from dataclasses import dataclass

from trajax.trajectory.common import Trajectory
from trajax.trajectory.basic import PathParameters, ReferencePoints

import numpy as np
from numtypes import Array, Dims, Dim1, D, shape_of
from scipy.interpolate import CubicSpline


type PointArray = Array[Dims[int, D[2]]]


@dataclass(kw_only=True, frozen=True)
class NumpyWaypointsTrajectory(Trajectory[PathParameters, ReferencePoints]):
    length: float
    reference_points: Array[Dim1]
    spline_x: CubicSpline
    spline_y: CubicSpline

    @staticmethod
    def create(*, points: PointArray, path_length: float) -> "NumpyWaypointsTrajectory":
        """Creates a waypoints trajectory from a set of 2D points.

        Args:
            points: Array of shape (N, 2) containing N waypoints with (x, y) coordinates.
            path_length: Total length of the trajectory.

        Returns:
            A waypoints trajectory using cubic spline interpolation.
        """
        x, y = points.T

        path_parameters = compute_path_parameters(x=x, y=y)
        length = path_parameters[-1]
        normalized_length = path_parameters * (path_length / length)

        spline_x = CubicSpline(normalized_length, x, bc_type="natural")
        spline_y = CubicSpline(normalized_length, y, bc_type="natural")

        return NumpyWaypointsTrajectory(
            length=path_length,
            reference_points=normalized_length,
            spline_x=spline_x,
            spline_y=spline_y,
        )

    def query[L: int, M: int](
        self, parameters: PathParameters[L, M]
    ) -> ReferencePoints[L, M]:
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

        return ReferencePoints.create(x=x, y=y, heading=heading)


def compute_path_parameters(*, x: Array[Dim1], y: Array[Dim1]) -> Array[Dim1]:
    segment_lengths = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])
