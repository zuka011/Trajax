from typing import overload, Literal
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    D_R,
    Trajectory,
    JaxPathParameters,
    JaxReferencePoints,
)
from trajax.trajectories.waypoints.basic import NumPyWaypointsTrajectory

from scipy.interpolate import CubicSpline
from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims, D

import jax
import jax.numpy as jnp
import numpy as np


type PointArray = Array[Dims[int, D[2]]]
type JaxPointArray = Float[JaxArray, "N 2"]


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxWaypointsTrajectory(Trajectory[JaxPathParameters, JaxReferencePoints]):
    length: float
    reference_points: Float[JaxArray, "N"]
    coefficients_x: Float[JaxArray, "N-1 4"]
    coefficients_y: Float[JaxArray, "N-1 4"]

    @overload
    @staticmethod
    def create(*, points: PointArray, path_length: float) -> "JaxWaypointsTrajectory":
        """Creates a JAX waypoints trajectory from a NumPy array."""
        ...

    @overload
    @staticmethod
    def create(
        *, points: JaxPointArray, path_length: float
    ) -> "JaxWaypointsTrajectory":
        """Creates a JAX waypoints trajectory from a JAX array."""
        ...

    @staticmethod
    def create(
        *, points: PointArray | JaxPointArray, path_length: float
    ) -> "JaxWaypointsTrajectory":
        """Creates a waypoints trajectory from a set of 2D points.

        Args:
            points: Array of shape (N, 2) containing N waypoints with (x, y) coordinates.
            path_length: Total length of the trajectory.

        Returns:
            A waypoints trajectory using cubic spline interpolation.
        """
        trajectory = NumPyWaypointsTrajectory.create(
            points=np.asarray(points), path_length=path_length
        )

        return JaxWaypointsTrajectory(
            length=trajectory.length,
            reference_points=jnp.asarray(trajectory.reference_points),
            coefficients_x=coefficients_from(trajectory.spline_x),
            coefficients_y=coefficients_from(trajectory.spline_y),
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

    @property
    def path_length(self) -> float:
        return self.length


def report_invalid_parameters(valid: bool, parameters: Float[JaxArray, "S 4"]) -> None:
    if not valid:
        print(f"Path parameters out of bounds. Got: {parameters}")


@jax.jit
@jaxtyped
def path_parameters_are_valid(
    path_parameters: Float[JaxArray, "T M"], *, path_length: Scalar
) -> Literal[True]:
    valid = jnp.all((0.0 <= path_parameters) & (path_parameters <= path_length))
    jax.debug.callback(report_invalid_parameters, valid, path_parameters)
    return True


def coefficients_from(spline: CubicSpline) -> Float[JaxArray, "S 4"]:
    """Extracts cubic spline coefficients from scipy CubicSpline.

    Returns coefficients [a, b, c, d] for each segment where:
    p(t) = a + b*t + c*t^2 + d*t^3
    with t being the local parameter within each segment.
    """
    # SciPy stores coefficients in shape (n_segments, 4) with columns [d, c, b, a]
    # We need to reorder to [a, b, c, d]
    # Transpose to get (n_segments, 4), Reverse to get [a, b, c, d]
    return jnp.asarray(spline.c.T[:, ::-1])


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

    return stack(x=x, y=y, heading=heading)


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
    b, c, d = (
        segment_coefficients[..., 1],
        segment_coefficients[..., 2],
        segment_coefficients[..., 3],
    )

    return b + 2 * c * t + 3 * d * t**2


@jax.jit
@jaxtyped
def stack(
    *,
    x: Float[JaxArray, "T M"],
    y: Float[JaxArray, "T M"],
    heading: Float[JaxArray, "T M"],
) -> Float[JaxArray, f"T {D_R} M"]:
    return jnp.stack([x, y, heading], axis=-1).transpose(0, 2, 1)
