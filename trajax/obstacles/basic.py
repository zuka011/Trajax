from typing import Sequence
from dataclasses import dataclass
from functools import cached_property

from trajax.types import DataType, D_o, D_O, SampledObstacleStates, ObstacleStates

from numtypes import Array, Dims, D, shape_of

import numpy as np


type ObstacleCovarianceArray[T: int = int, K: int = int] = Array[Dims[T, D_o, D_o, K]]


@dataclass(kw_only=True, frozen=True)
class NumPySampledObstacleStates[T: int, K: int, N: int](
    SampledObstacleStates[T, K, N]
):
    _x: Array[Dims[T, K, N]]
    _y: Array[Dims[T, K, N]]
    _heading: Array[Dims[T, K, N]]

    @staticmethod
    def create[T_: int, K_: int, N_: int](
        *,
        x: Array[Dims[T_, K_, N_]],
        y: Array[Dims[T_, K_, N_]],
        heading: Array[Dims[T_, K_, N_]],
    ) -> "NumPySampledObstacleStates[T_, K_, N_]":
        return NumPySampledObstacleStates(_x=x, _y=y, _heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        return self.array

    def x(self) -> Array[Dims[T, K, N]]:
        return self._x

    def y(self) -> Array[Dims[T, K, N]]:
        return self._y

    def heading(self) -> Array[Dims[T, K, N]]:
        return self._heading

    def at(self, *, time_step: int, sample: int) -> "NumPyObstacleStatesForTimeStep[K]":
        return NumPyObstacleStatesForTimeStep.create(
            x=self._x[time_step, :, sample],
            y=self._y[time_step, :, sample],
            heading=self._heading[time_step, :, sample],
        )

    @property
    def horizon(self) -> T:
        return self._x.shape[0]

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return self._x.shape[1]

    @property
    def sample_count(self) -> N:
        return self._x.shape[2]

    @property
    def array(self) -> Array[Dims[T, D_o, K, N]]:
        return self._array

    @cached_property
    def _array(self) -> Array[Dims[T, D_o, K, N]]:
        return np.stack([self._x, self._y, self._heading], axis=1)


@dataclass(kw_only=True, frozen=True)
class NumPyObstacle2dPositions[T: int, K: int]:
    _x: Array[Dims[T, K]]
    _y: Array[Dims[T, K]]

    @staticmethod
    def create[T_: int, K_: int](
        *, x: Array[Dims[T_, K_]], y: Array[Dims[T_, K_]]
    ) -> "NumPyObstacle2dPositions[T_, K_]":
        return NumPyObstacle2dPositions(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[2], K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self._x.shape[0]

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def count(self) -> K:
        return self._x.shape[1]

    @property
    def array(self) -> Array[Dims[T, D[2], K]]:
        return self._array

    @cached_property
    def _array(self) -> Array[Dims[T, D[2], K]]:
        return np.stack([self._x, self._y], axis=1)


@dataclass(kw_only=True, frozen=True)
class NumPyObstacleStates[T: int, K: int](
    ObstacleStates[T, K, NumPySampledObstacleStates[T, K, D[1]]]
):
    _x: Array[Dims[T, K]]
    _y: Array[Dims[T, K]]
    _heading: Array[Dims[T, K]]
    _covariance: ObstacleCovarianceArray[T, K] | None

    @staticmethod
    def empty[T_: int, K_: int = D[0]](
        *, horizon: T_, obstacle_count: K_ = 0
    ) -> "NumPyObstacleStates[T_, K_]":
        """Creates obstacle states for zero obstacles over the given time horizon."""
        empty = np.full((horizon, obstacle_count), fill_value=np.nan)

        assert shape_of(empty, matches=(horizon, obstacle_count))

        return NumPyObstacleStates.create(x=empty, y=empty, heading=empty)

    @staticmethod
    def wrap[T_: int, K_: int](
        array: Array[Dims[T_, D_o, K_]],
    ) -> "NumPyObstacleStates[T_, K_]":
        return NumPyObstacleStates.create(
            x=array[:, 0, :], y=array[:, 1, :], heading=array[:, 2, :]
        )

    @staticmethod
    def sampled[T_: int, K_: int, N_: int](  # type: ignore
        *,
        x: Array[Dims[T_, K_, N_]],
        y: Array[Dims[T_, K_, N_]],
        heading: Array[Dims[T_, K_, N_]],
    ) -> NumPySampledObstacleStates[T_, K_, N_]:
        return NumPySampledObstacleStates.create(x=x, y=y, heading=heading)

    @staticmethod
    def create[T_: int, K_: int](
        *,
        x: Array[Dims[T_, K_]],
        y: Array[Dims[T_, K_]],
        heading: Array[Dims[T_, K_]],
        covariance: ObstacleCovarianceArray[T_, K_] | None = None,
    ) -> "NumPyObstacleStates[T_, K_]":
        return NumPyObstacleStates(_x=x, _y=y, _heading=heading, _covariance=covariance)

    @staticmethod
    def of_states[T_: int, K_: int](
        obstacle_states: Sequence["NumPyObstacleStatesForTimeStep[K_]"],
        *,
        horizon: T_ | None = None,
    ) -> "NumPyObstacleStates[T_, K_]":
        assert len(obstacle_states) > 0, "Obstacle states sequence must not be empty."
        assert horizon is None or len(obstacle_states) == horizon, (
            f"Expected horizon {horizon}, but got {len(obstacle_states)} obstacle states."
        )

        x = np.stack([states.x() for states in obstacle_states], axis=0)
        y = np.stack([states.y() for states in obstacle_states], axis=0)
        heading = np.stack([states.heading() for states in obstacle_states], axis=0)

        return NumPyObstacleStates.create(x=x, y=y, heading=heading)

    @staticmethod
    def for_time_step[K_: int](
        *, x: Array[Dims[K_]], y: Array[Dims[K_]], heading: Array[Dims[K_]]
    ) -> "NumPyObstacleStatesForTimeStep[K_]":
        return NumPyObstacleStatesForTimeStep.create(x=x, y=y, heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        return self.array

    def x(self) -> Array[Dims[T, K]]:
        return self._x

    def y(self) -> Array[Dims[T, K]]:
        return self._y

    def heading(self) -> Array[Dims[T, K]]:
        return self._heading

    def positions(self) -> NumPyObstacle2dPositions[T, K]:
        return NumPyObstacle2dPositions.create(x=self._x, y=self._y)

    def covariance(self) -> ObstacleCovarianceArray[T, K] | None:
        return self._covariance

    def single(self) -> NumPySampledObstacleStates[T, K, D[1]]:
        return NumPySampledObstacleStates.create(
            x=self._x[..., np.newaxis],
            y=self._y[..., np.newaxis],
            heading=self._heading[..., np.newaxis],
        )

    def last(self) -> "NumPyObstacleStatesForTimeStep[K]":
        return self.at(time_step=self.horizon - 1)

    def at(self, *, time_step: int) -> "NumPyObstacleStatesForTimeStep[K]":
        return NumPyObstacleStatesForTimeStep.create(
            x=self._x[time_step],
            y=self._y[time_step],
            heading=self._heading[time_step],
        )

    @property
    def horizon(self) -> T:
        return self._x.shape[0]

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return self._x.shape[1]

    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
        return self._array

    @cached_property
    def _array(self) -> Array[Dims[T, D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=1)


@dataclass(kw_only=True, frozen=True)
class NumPyObstacle2dPositionsForTimeStep[K: int]:
    _x: Array[Dims[K]]
    _y: Array[Dims[K]]

    @staticmethod
    def create[K_: int](
        *, x: Array[Dims[K_]], y: Array[Dims[K_]]
    ) -> "NumPyObstacle2dPositionsForTimeStep[K_]":
        return NumPyObstacle2dPositionsForTimeStep(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D[2], K]]:
        return self.array

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def count(self) -> K:
        return self._x.shape[0]

    @property
    def array(self) -> Array[Dims[D[2], K]]:
        return self._array

    @cached_property
    def _array(self) -> Array[Dims[D[2], K]]:
        return np.stack([self._x, self._y], axis=0)


@dataclass(kw_only=True, frozen=True)
class NumPyObstacleStatesForTimeStep[K: int]:
    _x: Array[Dims[K]]
    _y: Array[Dims[K]]
    _heading: Array[Dims[K]]

    @staticmethod
    def create[K_: int](
        *,
        x: Array[Dims[K_]],
        y: Array[Dims[K_]],
        heading: Array[Dims[K_]],
    ) -> "NumPyObstacleStatesForTimeStep[K_]":
        return NumPyObstacleStatesForTimeStep(_x=x, _y=y, _heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_o, K]]:
        return self.array

    def x(self) -> Array[Dims[K]]:
        return self._x

    def y(self) -> Array[Dims[K]]:
        return self._y

    def heading(self) -> Array[Dims[K]]:
        return self._heading

    def positions(self) -> NumPyObstacle2dPositionsForTimeStep[K]:
        return NumPyObstacle2dPositionsForTimeStep.create(x=self._x, y=self._y)

    def replicate[T: int](self, *, horizon: T) -> NumPyObstacleStates[T, K]:
        return NumPyObstacleStates.create(
            x=np.tile(self._x[np.newaxis, :], (horizon, 1)),
            y=np.tile(self._y[np.newaxis, :], (horizon, 1)),
            heading=np.tile(self._heading[np.newaxis, :], (horizon, 1)),
        )

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return self._x.shape[0]

    @property
    def array(self) -> Array[Dims[D_o, K]]:
        return self._array

    @cached_property
    def _array(self) -> Array[Dims[D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=0)
