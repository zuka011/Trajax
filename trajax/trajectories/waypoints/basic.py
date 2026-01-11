import warnings
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    Trajectory,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositions,
    NumPyLateralPositions,
    NumPyLongitudinalPositions,
)

import numpy as np
from numtypes import Array, Dims, Dim1, D, shape_of
from scipy.interpolate import CubicSpline
from scipy.optimize import newton
from scipy.spatial import cKDTree  # type: ignore


type PointArray = Array[Dims[int, D[2]]]


@dataclass(kw_only=True, frozen=True)
class NumPyWaypointsTrajectory(
    Trajectory[
        NumPyPathParameters,
        NumPyReferencePoints,
        NumPyPositions,
        NumPyLateralPositions,
        NumPyLongitudinalPositions,
    ]
):
    length: float
    reference_points: Array[Dim1]
    spline_x: CubicSpline
    spline_y: CubicSpline

    kd_tree_samples: int
    refining_iterations: int

    @staticmethod
    def create(
        *,
        points: PointArray,
        path_length: float,
        kd_tree_samples: int = 200,
        refining_iterations: int = 3,
    ) -> "NumPyWaypointsTrajectory":
        """Creates a waypoints trajectory from a set of 2D points.

        Args:
            points: Array of shape (N, 2) containing N waypoints with (x, y) coordinates.
            path_length: Total length of the trajectory.
            kd_tree_samples: Number of samples to use for building the KD-tree for closest point search.
            refining_iterations: Number of iterations for local refinement of closest points.

        Returns:
            A waypoints trajectory using cubic spline interpolation.
        """
        x, y = points.T

        path_parameters = compute_path_parameters(x=x, y=y)
        length = path_parameters[-1]
        normalized_length = path_parameters * (path_length / length)

        spline_x = CubicSpline(normalized_length, x, bc_type="natural")
        spline_y = CubicSpline(normalized_length, y, bc_type="natural")

        return NumPyWaypointsTrajectory(
            length=path_length,
            reference_points=normalized_length,
            spline_x=spline_x,
            spline_y=spline_y,
            kd_tree_samples=kd_tree_samples,
            refining_iterations=refining_iterations,
        )

    def query[T: int, M: int](
        self, parameters: NumPyPathParameters[T, M]
    ) -> NumPyReferencePoints[T, M]:
        T, M = parameters.horizon, parameters.rollout_count
        s = np.asarray(parameters)

        assert np.all((0.0 <= s) & (s <= self.length)), (
            f"Path parameters out of bounds. Got: {s}"
        )

        x = self.spline_x(s).astype(np.float64, copy=False)
        y = self.spline_y(s).astype(np.float64, copy=False)

        dx_ds = self.spline_x(s, 1)
        dy_ds = self.spline_y(s, 1)
        heading = np.arctan2(dy_ds, dx_ds).astype(np.float64, copy=False)

        assert shape_of(x, matches=(T, M), name="x")
        assert shape_of(y, matches=(T, M), name="y")
        assert shape_of(heading, matches=(T, M), name="heading")

        return NumPyReferencePoints.create(x=x, y=y, heading=heading)

    def longitudinal[T: int, M: int](
        self, positions: NumPyPositions[T, M]
    ) -> NumPyLongitudinalPositions[T, M]:
        result = self._closest_arc_lengths(positions)
        return NumPyLongitudinalPositions.create(result)

    def lateral[T: int, M: int](
        self, positions: NumPyPositions[T, M]
    ) -> NumPyLateralPositions[T, M]:
        s = self._closest_arc_lengths(positions)

        closest_x = self.spline_x(s)
        closest_y = self.spline_y(s)
        dx_ds = self.spline_x(s, 1)
        dy_ds = self.spline_y(s, 1)
        tangent_norm = np.sqrt(dx_ds**2 + dy_ds**2)

        diff_x = positions.x() - closest_x
        diff_y = positions.y() - closest_y

        lateral = (diff_x * dy_ds - diff_y * dx_ds) / tangent_norm

        return NumPyLateralPositions.create(np.asarray(lateral, dtype=np.float64))

    @property
    def path_length(self) -> float:
        return self.length

    def _closest_arc_lengths[T: int, M: int](
        self, positions: NumPyPositions[T, M]
    ) -> Array[Dims[T, M]]:
        return self._refine(self._nearest_samples_to(positions), positions)

    def _nearest_samples_to[T: int, M: int](
        self, positions: NumPyPositions[T, M]
    ) -> Array[Dims[T, M]]:
        # Finds the nearest samples along the spline for given positions.
        T, M = positions.horizon, positions.rollout_count
        arc_lengths, tree = self._guess_samples

        query_points = positions.array.transpose(0, 2, 1).reshape(-1, 2)
        _, nearest_indices = tree.query(query_points)
        nearest_arc_lengths = arc_lengths[nearest_indices].reshape(T, M)

        assert shape_of(nearest_arc_lengths, matches=(T, M))

        return nearest_arc_lengths

    def _refine[T: int, M: int](
        self,
        arc_lengths: Array[Dims[T, M]],
        positions: NumPyPositions[T, M],
    ) -> Array[Dims[T, M]]:
        # Refines the arc lengths using Newton's method.
        s_0 = arc_lengths
        x, y = positions.x(), positions.y()

        def f(s):
            return (x - self.spline_x(s)) * self.spline_x(s, 1) + (
                y - self.spline_y(s)
            ) * self.spline_y(s, 1)

        def f_prime(s):
            c_x, c_y = self.spline_x(s), self.spline_y(s)
            dx, dy = self.spline_x(s, 1), self.spline_y(s, 1)
            ddx, ddy = self.spline_x(s, 2), self.spline_y(s, 2)
            return -(dx**2 + dy**2) + (x - c_x) * ddx + (y - c_y) * ddy

        # NOTE: Newton may not fully converge for points far from trajectory.
        # This is fine, since we don't care about high precision for such points.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="some failed to converge")
            result = newton(
                f, s_0, fprime=f_prime, maxiter=self.refining_iterations, disp=False
            )

        return np.clip(result, 0.0, self.length)

    @cached_property
    def _guess_samples(self) -> tuple[Array[Dim1], cKDTree]:
        # Computes samples along the spline and builds a KD-tree for fast lookup.
        s = np.linspace(0, self.length, self.kd_tree_samples)

        return s, cKDTree(np.column_stack([self.spline_x(s), self.spline_y(s)]))


def compute_path_parameters(*, x: Array[Dim1], y: Array[Dim1]) -> Array[Dim1]:
    segment_lengths = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])
