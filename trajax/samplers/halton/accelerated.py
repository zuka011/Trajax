from typing import Final, overload, cast
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    JaxControlInputBatchCreator,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxSampler,
)

from jaxtyping import Array as JaxArray, Float, Int, Scalar
from numtypes import Array, Dims

import jax
import jax.numpy as jnp
import jax.scipy.special as jspecial

type IntScalar = Int[JaxArray, ""]
type RadicalInverseCarry = tuple[
    IntScalar,  # current_index
    Scalar,  # result
    Scalar,  # factor
]

# fmt: off
PRIMES: Final = jnp.array([
2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
131, 137, 139, 149, 151, 157, 163, 167, 173,
])
# fmt: on

MAX_RADICAL_INVERSE_ITERATIONS: Final = 32


@dataclass(kw_only=True)
class JaxHaltonSplineSampler[
    BatchT: JaxControlInputBatch,
    D_u: int = int,
    M: int = int,
](JaxSampler[JaxControlInputSequence, BatchT]):
    standard_deviation: Final[Float[JaxArray, "D_u"]]
    to_batch: Final[JaxControlInputBatchCreator[BatchT]]
    knot_count: Final[int]
    halton_start_index: IntScalar

    _control_dimension: Final[D_u]
    _rollout_count: Final[M]

    @overload
    @staticmethod
    def create[B: JaxControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Array[Dims[D_u_]],
        rollout_count: M_,
        knot_count: int,
        to_batch: JaxControlInputBatchCreator[B],
        seed: int,
    ) -> "JaxHaltonSplineSampler[B, D_u_, M_]": ...

    @overload
    @staticmethod
    def create[B: JaxControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Float[JaxArray, "D_u"],
        control_dimension: D_u_ | None = None,
        rollout_count: M_,
        knot_count: int,
        to_batch: JaxControlInputBatchCreator[B],
        seed: int,
    ) -> "JaxHaltonSplineSampler[B, D_u_, M_]": ...

    @staticmethod
    def create[B: JaxControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Array[Dims[D_u_]] | Float[JaxArray, "D_u"],
        control_dimension: D_u_ | None = None,
        rollout_count: M_,
        knot_count: int,
        to_batch: JaxControlInputBatchCreator[B],
        seed: int,
    ) -> "JaxHaltonSplineSampler[B, D_u_, M_]":
        return JaxHaltonSplineSampler(
            standard_deviation=jnp.asarray(standard_deviation),
            to_batch=to_batch,
            knot_count=knot_count,
            halton_start_index=jnp.array(seed),
            _control_dimension=(
                control_dimension
                if control_dimension is not None
                else cast(D_u_, standard_deviation.shape[0])
            ),
            _rollout_count=rollout_count,
        )

    def __post_init__(self) -> None:
        assert self.knot_count >= 2, "Knot count must be at least 2."
        assert self.standard_deviation.shape[0] == self.control_dimension

    def sample(self, *, around: JaxControlInputSequence) -> BatchT:
        assert self.knot_count <= around.horizon, (
            f"Knot count ({self.knot_count}) cannot exceed time horizon ({around.horizon})."
        )

        samples, new_halton_index = sample_halton(
            around=around.array,
            standard_deviation=self.standard_deviation,
            rollout_count=self.rollout_count,
            knot_count=self.knot_count,
            halton_start_index=self.halton_start_index,
        )
        self.halton_start_index = new_halton_index
        return self.to_batch(array=samples)

    @property
    def control_dimension(self) -> D_u:
        return self._control_dimension

    @property
    def rollout_count(self) -> M:
        return self._rollout_count


@jax.jit(static_argnames=("rollout_count", "knot_count"))
@jaxtyped
def sample_halton(
    *,
    around: Float[JaxArray, "T D_u"],
    standard_deviation: Float[JaxArray, "D_u"],
    rollout_count: int,
    knot_count: int,
    halton_start_index: IntScalar,
) -> tuple[Float[JaxArray, "T D_u M"], IntScalar]:
    time_horizon, control_dimension = around.shape
    halton_dimensions = knot_count * control_dimension

    primes = PRIMES[:halton_dimensions]
    uniform_samples = generate_halton_sequence(
        primes, rollout_count, halton_start_index
    )

    gaussian_knots = transform_halton_to_knots(
        uniform_samples, knot_count, control_dimension
    )

    knot_times = jnp.linspace(0, time_horizon - 1, knot_count)
    evaluation_times = jnp.arange(time_horizon, dtype=jnp.float32)

    perturbations = interpolate_knots(gaussian_knots, knot_times, evaluation_times)
    scaled_perturbations = perturbations * standard_deviation[None, :, None]

    samples = around[..., None] + scaled_perturbations
    new_halton_index = halton_start_index + rollout_count

    return samples, new_halton_index


@jax.jit(static_argnames=("count",))
@jaxtyped
def generate_halton_sequence(
    primes: Int[JaxArray, "D"],
    count: int,
    start_index: IntScalar,
) -> Float[JaxArray, "count D"]:
    indices = jnp.arange(count) + start_index

    def compute_sample(index: Int[JaxArray, ""]) -> Float[JaxArray, "D"]:
        return jax.vmap(lambda base: radical_inverse(index, base))(primes)

    return jax.vmap(compute_sample)(indices)


@jax.jit(static_argnames=("knot_count", "control_dimension"))
@jaxtyped
def transform_halton_to_knots(
    uniform_samples: Float[JaxArray, "M halton_dim"],
    knot_count: int,
    control_dimension: int,
) -> Float[JaxArray, "M K D_u"]:
    gaussian_samples = inverse_normal_cdf(uniform_samples)
    return gaussian_samples.reshape(-1, knot_count, control_dimension)


@jax.jit
@jaxtyped
def radical_inverse(
    index: Int[JaxArray, ""], base: Int[JaxArray, ""]
) -> Float[JaxArray, ""]:
    def body_fn(
        carry: RadicalInverseCarry, _: None
    ) -> tuple[RadicalInverseCarry, None]:
        current_index, result, factor = carry
        digit = current_index % base
        result = result + digit * factor
        current_index = current_index // base
        factor = factor / base
        return (current_index, result, factor), None

    initial_state: RadicalInverseCarry = (index, 0.0, 1.0 / base)
    (_, result, _), _ = jax.lax.scan(
        body_fn, initial_state, None, length=MAX_RADICAL_INVERSE_ITERATIONS
    )

    return result


@jax.jit
@jaxtyped
def inverse_normal_cdf(
    uniform_values: Float[JaxArray, "*batch"],
) -> Float[JaxArray, "*batch"]:
    clamped = jnp.clip(uniform_values, 1e-7, 1 - 1e-7)
    return jnp.sqrt(2.0) * jspecial.erfinv(2.0 * clamped - 1.0)


@jax.jit
@jaxtyped
def solve_tridiagonal(
    lower: Float[JaxArray, "N"],
    diag: Float[JaxArray, "N"],
    upper: Float[JaxArray, "N"],
    rhs: Float[JaxArray, "N"],
) -> Float[JaxArray, "N"]:
    # NOTE: Thomas algorithm can be used, but the knot counts are small anyways.
    matrix = jnp.diag(diag) + jnp.diag(upper[:-1], k=1) + jnp.diag(lower[1:], k=-1)
    return jnp.linalg.solve(matrix, rhs)


@jax.jit
@jaxtyped
def compute_spline_coefficients(
    knot_times: Float[JaxArray, "K"],
    knot_values: Float[JaxArray, "K"],
) -> Float[JaxArray, "K"]:
    h = jnp.diff(knot_times)
    b = jnp.diff(knot_values) / h

    diag = jnp.concatenate([jnp.array([1.0]), 2 * (h[:-1] + h[1:]), jnp.array([1.0])])
    upper = jnp.concatenate([jnp.array([0.0]), h[1:], jnp.array([0.0])])
    lower = jnp.concatenate([jnp.array([0.0]), h[:-1], jnp.array([0.0])])
    rhs = jnp.concatenate([jnp.array([0.0]), 3 * (b[1:] - b[:-1]), jnp.array([0.0])])

    return solve_tridiagonal(lower, diag, upper, rhs)


@jax.jit
@jaxtyped
def evaluate_cubic_spline(
    knot_times: Float[JaxArray, "K"],
    knot_values: Float[JaxArray, "K"],
    second_derivatives: Float[JaxArray, "K"],
    evaluation_times: Float[JaxArray, "T"],
) -> Float[JaxArray, "T"]:
    knot_count = knot_times.shape[0]

    def evaluate_at_time(t: Float[JaxArray, ""]) -> Float[JaxArray, ""]:
        i = jnp.clip(
            jnp.searchsorted(knot_times, t, side="right") - 1, 0, knot_count - 2
        )

        t0, t1 = knot_times[i], knot_times[i + 1]
        y0, y1 = knot_values[i], knot_values[i + 1]
        d0, d1 = second_derivatives[i], second_derivatives[i + 1]

        h = t1 - t0
        a, b = t1 - t, t - t0

        return (
            (a / h) * y0
            + (b / h) * y1
            + ((a**3 / h - a * h) * d0 + (b**3 / h - b * h) * d1) / 6.0
        )

    return jax.vmap(evaluate_at_time)(evaluation_times)


@jax.jit
@jaxtyped
def interpolate_single_spline(
    knot_times: Float[JaxArray, "K"],
    knot_values: Float[JaxArray, "K"],
    evaluation_times: Float[JaxArray, "T"],
) -> Float[JaxArray, "T"]:
    second_derivatives = compute_spline_coefficients(knot_times, knot_values)
    return evaluate_cubic_spline(
        knot_times, knot_values, second_derivatives, evaluation_times
    )


@jax.jit
@jaxtyped
def interpolate_knots(
    gaussian_knots: Float[JaxArray, "M K D_u"],
    knot_times: Float[JaxArray, "K"],
    evaluation_times: Float[JaxArray, "T"],
) -> Float[JaxArray, "T D_u M"]:
    def interpolate_rollout(
        rollout_knots: Float[JaxArray, "K D_u"],
    ) -> Float[JaxArray, "T D_u"]:
        return jax.vmap(
            lambda knots: interpolate_single_spline(
                knot_times, knots, evaluation_times
            ),
            in_axes=1,
            out_axes=1,
        )(rollout_knots)

    perturbations = jax.vmap(interpolate_rollout)(gaussian_knots)
    return jnp.transpose(perturbations, (1, 2, 0))
