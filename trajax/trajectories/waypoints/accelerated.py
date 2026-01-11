from typing import overload, Literal, NamedTuple
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    D_R,
    Trajectory,
    JaxPathParameters,
    JaxReferencePoints,
    JaxPositions,
    JaxLateralPositions,
    JaxLongitudinalPositions,
)
from trajax.trajectories.waypoints.basic import NumPyWaypointsTrajectory

from scipy.interpolate import CubicSpline
from jaxtyping import Array as JaxArray, Float, Scalar, Int as JaxInt
from numtypes import Array, Dims, D

import jax
import jax.numpy as jnp
import numpy as np

type PointArray = Array[Dims[int, D[2]]]
type JaxPointArray = Float[JaxArray, "N 2"]
type Int = JaxInt[JaxArray, ""]


class GuessSamples(NamedTuple):
    x: Float[JaxArray, "K"]
    y: Float[JaxArray, "K"]
    arc_lengths: Float[JaxArray, "K"]


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxWaypointsTrajectory(
    Trajectory[
        JaxPathParameters,
        JaxReferencePoints,
        JaxPositions,
        JaxLateralPositions,
        JaxLongitudinalPositions,
    ]
):
    length: Scalar
    reference_points: Float[JaxArray, "N"]
    coefficients_x: Float[JaxArray, "N-1 4"]
    coefficients_y: Float[JaxArray, "N-1 4"]

    guess_samples: GuessSamples
    refining_iterations: int

    @overload
    @staticmethod
    def create(
        *,
        points: PointArray,
        path_length: float,
        coarse_samples: int = 200,
        refining_iterations: int = 3,
    ) -> "JaxWaypointsTrajectory":
        """Creates a waypoints trajectory from a set of 2D points."""
        ...

    @overload
    @staticmethod
    def create(
        *,
        points: JaxPointArray,
        path_length: float,
        coarse_samples: int = 200,
        refining_iterations: int = 3,
    ) -> "JaxWaypointsTrajectory":
        """Creates a waypoints trajectory from a set of 2D points."""
        ...

    @staticmethod
    def create(
        *,
        points: PointArray | JaxPointArray,
        path_length: float,
        coarse_samples: int = 200,
        refining_iterations: int = 3,
    ) -> "JaxWaypointsTrajectory":
        """Creates a waypoints trajectory from a set of 2D points."""
        trajectory = NumPyWaypointsTrajectory.create(
            points=np.asarray(points), path_length=path_length
        )

        return JaxWaypointsTrajectory(
            length=jnp.asarray(trajectory.length),
            reference_points=jnp.asarray(trajectory.reference_points),
            coefficients_x=coefficients_from(trajectory.spline_x),
            coefficients_y=coefficients_from(trajectory.spline_y),
            guess_samples=compute_guess_samples(
                trajectory=trajectory, sample_count=coarse_samples
            ),
            refining_iterations=refining_iterations,
        )

    def query[T: int, M: int](
        self, parameters: JaxPathParameters[T, M]
    ) -> JaxReferencePoints[T, M]:
        assert path_parameters_are_valid(parameters.array, path_length=self.path_length)

        return JaxReferencePoints(
            query(
                parameters=parameters.array,
                reference_points=self.reference_points,
                coefficients_x=self.coefficients_x,
                coefficients_y=self.coefficients_y,
            )
        )

    def lateral[T: int, M: int](
        self, positions: JaxPositions[T, M]
    ) -> JaxLateralPositions[T, M]:
        return JaxLateralPositions(
            compute_lateral(
                x=positions.x_array,
                y=positions.y_array,
                arc_lengths=closest_arc_lengths(
                    x=positions.x_array,
                    y=positions.y_array,
                    reference_points=self.reference_points,
                    coefficients_x=self.coefficients_x,
                    coefficients_y=self.coefficients_y,
                    guess=self.guess_samples,
                    path_length=self.path_length,
                    refining_iterations=self.refining_iterations,
                ),
                reference_points=self.reference_points,
                coefficients_x=self.coefficients_x,
                coefficients_y=self.coefficients_y,
            )
        )

    def longitudinal[T: int, M: int](
        self, positions: JaxPositions[T, M]
    ) -> JaxLongitudinalPositions[T, M]:
        return JaxLongitudinalPositions(
            closest_arc_lengths(
                x=positions.x_array,
                y=positions.y_array,
                reference_points=self.reference_points,
                coefficients_x=self.coefficients_x,
                coefficients_y=self.coefficients_y,
                guess=self.guess_samples,
                path_length=self.path_length,
                refining_iterations=self.refining_iterations,
            )
        )

    @property
    def path_length(self) -> float:
        return float(self.length)


def report_invalid_parameters(valid: bool, parameters: Float[JaxArray, "S 4"]) -> None:
    if not valid:
        print(f"Path parameters out of bounds. Got: {parameters}")


def coefficients_from(spline: CubicSpline) -> Float[JaxArray, "S 4"]:
    # Extracts cubic spline coefficients from scipy CubicSpline.
    #
    # Returns coefficients [a, b, c, d] for each segment where:
    # p(t) = a + b*t + c*t^2 + d*t^3
    # with t being the local parameter within each segment.

    # SciPy stores coefficients in shape (n_segments, 4) with columns [d, c, b, a]
    # We need to reorder to [a, b, c, d]
    # Transpose to get (n_segments, 4), Reverse to get [a, b, c, d]
    return jnp.asarray(spline.c.T[:, ::-1])


def compute_guess_samples(
    trajectory: NumPyWaypointsTrajectory, *, sample_count: int
) -> GuessSamples:
    sample_s = np.linspace(0, trajectory.length, sample_count)
    sample_x = trajectory.spline_x(sample_s)
    sample_y = trajectory.spline_y(sample_s)

    return GuessSamples(
        x=jnp.asarray(sample_x),
        y=jnp.asarray(sample_y),
        arc_lengths=jnp.asarray(sample_s),
    )


@jax.jit
@jaxtyped
def path_parameters_are_valid(
    path_parameters: Float[JaxArray, "T M"], *, path_length: Scalar
) -> Literal[True]:
    valid = jnp.all((0.0 <= path_parameters) & (path_parameters <= path_length))
    jax.debug.callback(report_invalid_parameters, valid, path_parameters)
    return True


@jax.jit
@jaxtyped
def query(
    parameters: Float[JaxArray, "T M"],
    *,
    reference_points: Float[JaxArray, "N"],
    coefficients_x: Float[JaxArray, "S 4"],
    coefficients_y: Float[JaxArray, "S 4"],
) -> Float[JaxArray, f"T {D_R} M"]:
    x = evaluate(parameters, reference_points, coefficients_x)
    y = evaluate(parameters, reference_points, coefficients_y)

    dx_ds = evaluate_derivative(parameters, reference_points, coefficients_x)
    dy_ds = evaluate_derivative(parameters, reference_points, coefficients_y)
    heading = jnp.arctan2(dy_ds, dx_ds)

    return jnp.stack([x, y, heading], axis=1)


@jax.jit
@jaxtyped
def evaluate(
    parameters: Float[JaxArray, "T M"],
    reference_points: Float[JaxArray, "N"],
    coefficients: Float[JaxArray, "S 4"],
) -> Float[JaxArray, "T M"]:
    num_segments = len(coefficients)
    segment_indices = jnp.clip(
        jnp.searchsorted(reference_points[1:], parameters, side="right"),
        0,
        num_segments - 1,
    )
    segment_starts = reference_points[segment_indices]
    t = parameters - segment_starts

    segment_coefficients = coefficients[segment_indices]
    a, b, c, d = (
        segment_coefficients[..., 0],
        segment_coefficients[..., 1],
        segment_coefficients[..., 2],
        segment_coefficients[..., 3],
    )

    return a + b * t + c * t**2 + d * t**3


@jax.jit
@jaxtyped
def evaluate_derivative(
    parameters: Float[JaxArray, "T M"],
    reference_points: Float[JaxArray, "N"],
    coefficients: Float[JaxArray, "S 4"],
) -> Float[JaxArray, "T M"]:
    num_segments = len(coefficients)
    segment_indices = jnp.clip(
        jnp.searchsorted(reference_points[1:], parameters, side="right"),
        0,
        num_segments - 1,
    )
    segment_starts = reference_points[segment_indices]
    t = parameters - segment_starts

    segment_coefficients = coefficients[segment_indices]
    b = segment_coefficients[..., 1]
    c = segment_coefficients[..., 2]
    d = segment_coefficients[..., 3]

    return b + 2 * c * t + 3 * d * t**2


@jax.jit
@jaxtyped
def evaluate_second_derivative(
    parameters: Float[JaxArray, "T M"],
    reference_points: Float[JaxArray, "N"],
    coefficients: Float[JaxArray, "S 4"],
) -> Float[JaxArray, "T M"]:
    num_segments = len(coefficients)
    segment_indices = jnp.clip(
        jnp.searchsorted(reference_points[1:], parameters, side="right"),
        0,
        num_segments - 1,
    )
    segment_starts = reference_points[segment_indices]
    t = parameters - segment_starts

    segment_coefficients = coefficients[segment_indices]
    c = segment_coefficients[..., 2]
    d = segment_coefficients[..., 3]

    return 2 * c + 6 * d * t


@jax.jit(static_argnames=("refining_iterations",))
@jaxtyped
def closest_arc_lengths(
    *,
    x: Float[JaxArray, "T M"],
    y: Float[JaxArray, "T M"],
    reference_points: Float[JaxArray, "N"],
    coefficients_x: Float[JaxArray, "S 4"],
    coefficients_y: Float[JaxArray, "S 4"],
    guess: GuessSamples,
    path_length: Scalar,
    refining_iterations: int,
) -> Float[JaxArray, "T M"]:
    return refine_newton(
        arc_lengths=nearest_sample(x=x, y=y, guess=guess),
        x=x,
        y=y,
        reference_points=reference_points,
        coefficients_x=coefficients_x,
        coefficients_y=coefficients_y,
        path_length=path_length,
        iterations=refining_iterations,
    )


@jax.jit
@jaxtyped
def nearest_sample(
    *,
    x: Float[JaxArray, "T M"],
    y: Float[JaxArray, "T M"],
    guess: GuessSamples,
) -> Float[JaxArray, "T M"]:
    # Find nearest pre-computed sample for each position (coarse search).
    diff_x = guess.x[:, None, None] - x[None, :, :]
    diff_y = guess.y[:, None, None] - y[None, :, :]
    distances_sq = diff_x**2 + diff_y**2

    nearest_idx = jnp.argmin(distances_sq, axis=0)
    return guess.arc_lengths[nearest_idx]


@jax.jit(static_argnames=("iterations",))
@jaxtyped
def refine_newton(
    *,
    arc_lengths: Float[JaxArray, "T M"],
    x: Float[JaxArray, "T M"],
    y: Float[JaxArray, "T M"],
    reference_points: Float[JaxArray, "N"],
    coefficients_x: Float[JaxArray, "S 4"],
    coefficients_y: Float[JaxArray, "S 4"],
    path_length: Scalar,
    iterations: int,
) -> Float[JaxArray, "T M"]:
    # Refine arc length estimates using Newton's method.

    def newton_step(
        s: Float[JaxArray, "T M"], _: None
    ) -> tuple[Float[JaxArray, "T M"], None]:
        cx = evaluate(s, reference_points, coefficients_x)
        cy = evaluate(s, reference_points, coefficients_y)

        dx = evaluate_derivative(s, reference_points, coefficients_x)
        dy = evaluate_derivative(s, reference_points, coefficients_y)

        ddx = evaluate_second_derivative(s, reference_points, coefficients_x)
        ddy = evaluate_second_derivative(s, reference_points, coefficients_y)

        # f(s) = (p - c(s)) · c'(s) = 0  (orthogonality condition)
        f = (x - cx) * dx + (y - cy) * dy

        # f'(s) = -|c'(s)|² + (p - c(s)) · c''(s)
        f_prime = -(dx**2 + dy**2) + (x - cx) * ddx + (y - cy) * ddy

        # Newton step
        s_new = s - f / f_prime
        s_new = jnp.clip(s_new, 0.0, path_length)

        return s_new, None

    refined, _ = jax.lax.scan(newton_step, arc_lengths, None, length=iterations)
    return refined


@jax.jit
@jaxtyped
def compute_lateral(
    *,
    x: Float[JaxArray, "T M"],
    y: Float[JaxArray, "T M"],
    arc_lengths: Float[JaxArray, "T M"],
    reference_points: Float[JaxArray, "N"],
    coefficients_x: Float[JaxArray, "S 4"],
    coefficients_y: Float[JaxArray, "S 4"],
) -> Float[JaxArray, "T M"]:
    # Compute signed lateral deviation from trajectory.
    closest_x = evaluate(arc_lengths, reference_points, coefficients_x)
    closest_y = evaluate(arc_lengths, reference_points, coefficients_y)

    dx_ds = evaluate_derivative(arc_lengths, reference_points, coefficients_x)
    dy_ds = evaluate_derivative(arc_lengths, reference_points, coefficients_y)
    tangent_norm = jnp.sqrt(dx_ds**2 + dy_ds**2)

    diff_x = x - closest_x
    diff_y = y - closest_y

    # NOTE: Cross product for signed lateral (right = positive, left = negative)
    lateral = (diff_x * dy_ds - diff_y * dx_ds) / tangent_norm

    return lateral
