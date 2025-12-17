from dataclasses import dataclass

from trajax.trajectory.common import Trajectory
from trajax.trajectory.basic import NumPyPathParameters, NumPyReferencePoints

from numtypes import shape_of

import numpy as np


@dataclass(kw_only=True, frozen=True)
class NumpyLineTrajectory(Trajectory[NumPyPathParameters, NumPyReferencePoints]):
    start: tuple[float, float]
    end: tuple[float, float]

    delta_x: float
    delta_y: float
    length: float
    heading: float

    @staticmethod
    def create(
        *, start: tuple[float, float], end: tuple[float, float], path_length: float
    ) -> "NumpyLineTrajectory":
        """Generates a straight line trajectory from start to end."""
        return NumpyLineTrajectory(
            start=start,
            end=end,
            delta_x=(delta_x := end[0] - start[0]),
            delta_y=(delta_y := end[1] - start[1]),
            length=path_length,
            heading=np.arctan2(delta_y, delta_x),
        )

    def query[T: int, M: int](
        self, parameters: NumPyPathParameters[T, M]
    ) -> NumPyReferencePoints[T, M]:
        T, M = parameters.horizon, parameters.rollout_count
        normalized = np.asarray(parameters) / self.length
        x = self.start[0] + normalized * self.delta_x
        y = self.start[1] + normalized * self.delta_y
        heading = np.full_like(x, self.heading)

        assert shape_of(x, matches=(T, M), name="x")
        assert shape_of(y, matches=(T, M), name="y")
        assert shape_of(heading, matches=(T, M), name="heading")

        return NumPyReferencePoints.create(x=x, y=y, heading=heading)
