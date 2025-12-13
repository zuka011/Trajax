from dataclasses import dataclass

from trajax.trajectory.common import D_r, D_R

import numpy as np
from numtypes import array, Array, Dims, shape_of


@dataclass(frozen=True)
class PathParameters[T: int, M: int]:
    array: Array[Dims[T, M]]

    def __array__(self) -> Array[Dims[T, M]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[1]


@dataclass(frozen=True)
class Positions[T: int, M: int]:
    x: Array[Dims[T, M]]
    y: Array[Dims[T, M]]

    @staticmethod
    def create[T_: int, M_: int](
        *,
        x: Array[Dims[T_, M_]],
        y: Array[Dims[T_, M_]],
    ) -> "Positions[T_, M_]":
        """Creates a NumPy positions instance from x and y coordinate arrays."""
        return Positions(x=x, y=y)


@dataclass(frozen=True)
class ReferencePoints[T: int, M: int]:
    array: Array[Dims[T, D_r, M]]

    @staticmethod
    def create[T_: int, M_: int](
        *,
        x: Array[Dims[T_, M_]],
        y: Array[Dims[T_, M_]],
        heading: Array[Dims[T_, M_]],
    ) -> "ReferencePoints[T_, M_]":
        """Creates a NumPy reference points instance from x, y, and heading arrays."""
        T, M = x.shape
        return ReferencePoints(
            array=array(
                np.stack([x, y, heading], axis=-1).transpose(0, 2, 1).tolist(),
                shape=(T, D_R, M),
            )
        )

    def __array__(self) -> Array[Dims[T, D_r, M]]:
        return self.array

    @property
    def x(self) -> Array[Dims[T, M]]:
        return self.array[:, 0]

    @property
    def y(self) -> Array[Dims[T, M]]:
        return self.array[:, 1]

    @property
    def heading(self) -> Array[Dims[T, M]]:
        return self.array[:, 2]


@dataclass(kw_only=True, frozen=True)
class NumpyLineTrajectory:
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

    def query[L: int, M: int](
        self, parameters: PathParameters[L, M]
    ) -> ReferencePoints[L, M]:
        T, M = parameters.horizon, parameters.rollout_count
        normalized = np.asarray(parameters) / self.length
        x = self.start[0] + normalized * self.delta_x
        y = self.start[1] + normalized * self.delta_y
        heading = np.full_like(x, self.heading)

        assert shape_of(x, matches=(T, M), name="x")
        assert shape_of(y, matches=(T, M), name="y")
        assert shape_of(heading, matches=(T, M), name="heading")

        return ReferencePoints.create(x=x, y=y, heading=heading)
