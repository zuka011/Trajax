from typing import Final, overload, cast
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    JaxControlInputBatchCreator,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxSampler,
)

from jaxtyping import Array as JaxArray, Float, PRNGKeyArray
from numtypes import Array, Dims

import jax
import jax.random as jrandom
import jax.numpy as jnp


@dataclass(kw_only=True)
class JaxGaussianSampler[BatchT: JaxControlInputBatch, D_u: int = int, M: int = int](
    JaxSampler[JaxControlInputSequence, BatchT]
):
    standard_deviation: Final[Float[JaxArray, "D_u"]]
    to_batch: Final[JaxControlInputBatchCreator[BatchT]]

    _control_dimension: Final[D_u]
    _rollout_count: Final[M]

    key: PRNGKeyArray

    @overload
    @staticmethod
    def create[B: JaxControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Array[Dims[D_u_]],
        rollout_count: M_,
        to_batch: JaxControlInputBatchCreator[B],
        key: PRNGKeyArray | None = None,
        seed: int | None = None,
    ) -> "JaxGaussianSampler[B, D_u_, M_]":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        ...

    @overload
    @staticmethod
    def create[B: JaxControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Float[JaxArray, "D_u"],
        control_dimension: D_u_ | None = None,
        rollout_count: M_,
        to_batch: JaxControlInputBatchCreator[B],
        key: PRNGKeyArray | None = None,
        seed: int | None = None,
    ) -> "JaxGaussianSampler[B, D_u_, M_]":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        ...

    @staticmethod
    def create[B: JaxControlInputBatch, D_u_: int, M_: int](
        *,
        standard_deviation: Array[Dims[D_u_]] | Float[JaxArray, "D_u"],
        control_dimension: D_u_ | None = None,
        rollout_count: M_,
        to_batch: JaxControlInputBatchCreator[B],
        key: PRNGKeyArray | None = None,
        seed: int | None = None,
    ) -> "JaxGaussianSampler[B, D_u_, M_]":
        """Creates a sampler generating Gaussian noise around the specified control input
        sequence.
        """
        return JaxGaussianSampler(
            standard_deviation=jnp.asarray(standard_deviation),
            to_batch=to_batch,
            _control_dimension=(
                control_dimension
                if control_dimension is not None
                else cast(D_u_, standard_deviation.shape[0])
            ),
            _rollout_count=rollout_count,
            key=key if key is not None else jrandom.PRNGKey(seed or 0),
        )

    def __post_init__(self) -> None:
        assert self.standard_deviation.shape[0] == self.control_dimension, (
            f"Expected standard deviation with shape ({self.control_dimension},), "
            f"but got {self.standard_deviation.shape}"
        )

    def sample(self, *, around: JaxControlInputSequence) -> BatchT:
        self.key, samples = sample(
            self.key,
            around=around.array,
            standard_deviation=self.standard_deviation,
            rollout_count=self.rollout_count,
        )

        return self.to_batch(array=samples)

    @property
    def control_dimension(self) -> D_u:
        return self._control_dimension

    @property
    def rollout_count(self) -> M:
        return self._rollout_count


@jax.jit(static_argnames=("rollout_count",))
@jaxtyped
def sample(
    key: PRNGKeyArray,
    *,
    around: Float[JaxArray, "T D_u"],
    standard_deviation: Float[JaxArray, "D_u"],
    rollout_count: int,
) -> tuple[PRNGKeyArray, Float[JaxArray, "T D_u M"]]:
    time_horizon, control_dimension = around.shape

    key, subkey = jrandom.split(key)
    samples = around[..., None] + standard_deviation[None, :, None] * jrandom.normal(
        subkey, shape=(time_horizon, control_dimension, rollout_count)
    )

    return key, samples
