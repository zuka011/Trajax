from typing import Sequence, Self, Any, cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    DataType,
    D_o,
    D_O,
    SampledObstacleStates,
    ObstacleStates,
    NumPyObstacleStateProvider,
    ObstacleMotionPredictor,
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


@dataclass(kw_only=True, frozen=True)
class NumPyObstacleStatesRunningHistory[K: int]:
    history: list[NumPyObstacleStatesForTimeStep[K]]

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

    def append(
        self, step: NumPyObstacleStatesForTimeStep[K]
    ) -> "NumPyObstacleStatesRunningHistory[K]":
        return NumPyObstacleStatesRunningHistory(history=self.history + [step])

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

    @cached_property
    def array(self) -> Array[Dims[int, D_o, K]]:
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


@dataclass(frozen=True)
class NumPyStaticObstacleStateProvider[T: int, K: int](
    NumPyObstacleStateProvider[NumPyObstacleStates[T, K]]
):
    positions: NumPyObstacleStates[T, K]

    @staticmethod
    def empty[T_: int](*, horizon: T_) -> "NumPyStaticObstacleStateProvider[T_, D[0]]":
        positions = np.empty((0, 2))

        assert shape_of(positions, matches=(0, 2))

        return NumPyStaticObstacleStateProvider.create(
            positions=positions, horizon=horizon
        )

    @staticmethod
    def create[T_: int, K_: int](
        *,
        positions: Array[Dims[K_, D[2]]],
        headings: Array[Dims[K_]] | None = None,
        horizon: T_,
    ) -> "NumPyStaticObstacleStateProvider[T_, K_]":
        K = positions.shape[0]
        x = np.tile(positions[:, 0], (horizon, 1))
        y = np.tile(positions[:, 1], (horizon, 1))

        if headings is not None:
            heading = np.tile(headings, (horizon, 1))
        else:
            heading = np.zeros((horizon, K))

        assert shape_of(x, matches=(horizon, K))
        assert shape_of(y, matches=(horizon, K))
        assert shape_of(heading, matches=(horizon, K))

        return NumPyStaticObstacleStateProvider(
            NumPyObstacleStates.create(x=x, y=y, heading=heading)
        )

    def with_time_step(self, time_step: float) -> Self:
        # Time step does not matter.
        return self

    def with_predictor(self, predictor: Any) -> Self:
        # Predictor does not matter.
        return self

    def __call__(self) -> NumPyObstacleStates[T, K]:
        return self.positions

    def step(self) -> None:
        # Nothing to do, since the obstacles are static.
        pass


@dataclass(kw_only=True)
class NumPyDynamicObstacleStateProvider[T: int, K: int](
    NumPyObstacleStateProvider[NumPyObstacleStates[T, K]]
):
    type MotionPredictor[T_: int, K_: int] = ObstacleMotionPredictor[
        NumPyObstacleStatesRunningHistory[K_], NumPyObstacleStates[T_, K_]
    ]

    history: NumPyObstacleStatesRunningHistory[K]
    velocities: Array[Dims[K, D[2]]]

    horizon: T
    time_step: float | None = None
    predictor: MotionPredictor | None = None

    @staticmethod
    def create[T_: int, K_: int](
        *,
        positions: Array[Dims[K_, D[2]]],
        velocities: Array[Dims[K_, D[2]]],
        horizon: T_,
    ) -> "NumPyDynamicObstacleStateProvider[T_, K_]":
        headings = headings_from(velocities)

        return NumPyDynamicObstacleStateProvider(
            history=NumPyObstacleStatesRunningHistory.single(
                NumPyObstacleStatesForTimeStep.create(
                    x=positions[:, 0], y=positions[:, 1], heading=headings
                )
            ),
            velocities=velocities,
            horizon=horizon,
        )

    def with_time_step(self, time_step: float) -> Self:
        return self.__class__(
            history=self.history,
            velocities=self.velocities,
            predictor=self.predictor,
            horizon=self.horizon,
            time_step=time_step,
        )

    def with_predictor(self, predictor: MotionPredictor) -> Self:
        return self.__class__(
            history=self.history,
            velocities=self.velocities,
            predictor=predictor,
            horizon=self.horizon,
            time_step=self.time_step,
        )

    def __call__(self) -> NumPyObstacleStates[T, K]:
        assert self.predictor is not None, (
            "Motion predictor must be set to provide obstacle states."
        )

        estimated_states = self.predictor.predict(history=self.history)

        return estimated_states

    def step(self) -> None:
        assert self.time_step is not None, (
            "Time step must be set to advance obstacle states."
        )

        last = self.history.last()
        new = NumPyObstacleStatesForTimeStep.create(
            x=last._x + self.velocities[:, 0] * self.time_step,
            y=last._y + self.velocities[:, 1] * self.time_step,
            heading=last._heading,
        )
        self.history = self.history.append(new)


def headings_from[K: int](velocities: Array[Dims[K, D[2]]]) -> Array[Dims[K]]:
    K = velocities.shape[0]
    speed = np.linalg.norm(velocities, axis=1)
    heading = np.where(
        speed > 1e-6,
        np.arctan2(velocities[:, 1], velocities[:, 0]),
        0.0,
    )

    assert shape_of(heading, matches=(K,))

    return heading
