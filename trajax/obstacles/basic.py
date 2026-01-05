from typing import Sequence, Self, cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    DataType,
    D_o,
    D_O,
    SampledObstacleStates,
    ObstacleStates,
)

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
        return np.stack([self._x, self._y, self._heading], axis=1)

    def x(self) -> Array[Dims[T, K, N]]:
        return self._x

    def y(self) -> Array[Dims[T, K, N]]:
        return self._y

    def heading(self) -> Array[Dims[T, K, N]]:
        return self._heading


@dataclass(kw_only=True, frozen=True)
class NumPyObstacleStates[T: int, K: int](
    ObstacleStates[T, K, NumPySampledObstacleStates[T, K, D[1]]]
):
    _x: Array[Dims[T, K]]
    _y: Array[Dims[T, K]]
    _heading: Array[Dims[T, K]]
    _covariance: ObstacleCovarianceArray[T, K] | None

    @staticmethod
    def empty[T_: int](*, horizon: T_) -> "NumPyObstacleStates[T_, D[0]]":
        """Creates obstacle states for zero obstacles over the given time horizon."""
        empty = np.empty((horizon, 0))

        assert shape_of(empty, matches=(horizon, 0))

        return NumPyObstacleStates.create(x=empty, y=empty, heading=empty)

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
        obstacle_states: Sequence["NumPyObstacleStates[int, K_]"],
        *,
        horizon: T_ | None = None,
    ) -> "NumPyObstacleStates[T_, K_]":
        assert horizon is None or len(obstacle_states) == horizon, (
            f"Expected horizon {horizon}, but got {len(obstacle_states)} obstacle states."
        )

        x = np.stack([states.x()[0] for states in obstacle_states], axis=0)
        y = np.stack([states.y()[0] for states in obstacle_states], axis=0)
        heading = np.stack([states.heading()[0] for states in obstacle_states], axis=0)

        return NumPyObstacleStates.create(x=x, y=y, heading=heading)

    @staticmethod
    def for_time_step[K_: int](
        *, x: Array[Dims[K_]], y: Array[Dims[K_]], heading: Array[Dims[K_]]
    ) -> "NumPyObstacleStatesForTimeStep[K_]":
        return NumPyObstacleStatesForTimeStep.create(x=x, y=y, heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=1)

    def x(self) -> Array[Dims[T, K]]:
        return self._x

    def y(self) -> Array[Dims[T, K]]:
        return self._y

    def heading(self) -> Array[Dims[T, K]]:
        return self._heading

    def covariance(self) -> ObstacleCovarianceArray[T, K] | None:
        return self._covariance

    def single(self) -> NumPySampledObstacleStates[T, K, D[1]]:
        return NumPySampledObstacleStates.create(
            x=self._x[..., np.newaxis],
            y=self._y[..., np.newaxis],
            heading=self._heading[..., np.newaxis],
        )

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
        return np.stack([self._x, self._y, self._heading], axis=1)


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

    @property
    def x(self) -> Array[Dims[K]]:
        return self._x

    @property
    def y(self) -> Array[Dims[K]]:
        return self._y

    @property
    def heading(self) -> Array[Dims[K]]:
        return self._heading


@dataclass(kw_only=True, frozen=True)
class NumPyObstacleStatesRunningHistory[K: int]:
    history: list[NumPyObstacleStatesForTimeStep[K]]

    @staticmethod
    def empty() -> "NumPyObstacleStatesRunningHistory[int]":
        return NumPyObstacleStatesRunningHistory(history=[])

    @staticmethod
    def single[K_: int](
        step: NumPyObstacleStatesForTimeStep[K_],
    ) -> "NumPyObstacleStatesRunningHistory[K_]":
        return NumPyObstacleStatesRunningHistory(history=[step])

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[int, D_o, K]]:
        return self.array

    def last(self) -> NumPyObstacleStatesForTimeStep[K]:
        assert self.horizon > 0, "Cannot get last state from empty history."

        return self.history[-1]

    def append(self, step: NumPyObstacleStatesForTimeStep[K]) -> Self:
        return self.__class__(history=self.history + [step])

    def get(self) -> Self:
        return self

    def x(self) -> Array[Dims[int, K]]:
        return self._x

    def y(self) -> Array[Dims[int, K]]:
        return self._y

    def heading(self) -> Array[Dims[int, K]]:
        return self._heading

    @property
    def horizon(self) -> int:
        return len(self.history)

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        if self.horizon == 0:
            return cast(K, 0)

        return self.history[0]._x.shape[0]

    @property
    def array(self) -> Array[Dims[int, D_o, K]]:
        return self._array

    @cached_property
    def _array(self) -> Array[Dims[int, D_o, K]]:
        return np.stack([self.x(), self.y(), self.heading()], axis=1)

    @cached_property
    def _x(self) -> Array[Dims[int, K]]:
        return np.stack([step._x for step in self.history], axis=0)

    @cached_property
    def _y(self) -> Array[Dims[int, K]]:
        return np.stack([step._y for step in self.history], axis=0)

    @cached_property
    def _heading(self) -> Array[Dims[int, K]]:
        return np.stack([step._heading for step in self.history], axis=0)
