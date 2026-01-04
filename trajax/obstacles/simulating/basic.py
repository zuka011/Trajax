from typing import Self
from dataclasses import dataclass

from trajax.types import NumPyObstacleStateProvider, ObstacleMotionPredictor
from trajax.obstacles.basic import (
    NumPyObstacleStates,
    NumPyObstacleStatesForTimeStep,
    NumPyObstacleStatesRunningHistory,
)


from numtypes import Array, Dims, D, shape_of

import numpy as np


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
    predictor: MotionPredictor[T, K] | None = None

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

        return self.predictor.predict(history=self.history)

    def step(self) -> None:
        assert self.time_step is not None, (
            "Time step must be set to advance obstacle states."
        )

        last = self.history.last()
        x, y = step_obstacles(
            x=last.x, y=last.y, velocities=self.velocities, time_step=self.time_step
        )

        self.history = self.history.append(
            NumPyObstacleStatesForTimeStep.create(x=x, y=y, heading=last.heading)
        )


def step_obstacles[K: int](
    *,
    x: Array[Dims[K]],
    y: Array[Dims[K]],
    velocities: Array[Dims[K, D[2]]],
    time_step: float,
) -> tuple[Array[Dims[K]], Array[Dims[K]]]:
    K = x.shape[0]
    new_x = x + velocities[:, 0] * time_step
    new_y = y + velocities[:, 1] * time_step

    assert shape_of(new_x, matches=(K,))
    assert shape_of(new_y, matches=(K,))

    return new_x, new_y


def headings_from[K: int](velocities: Array[Dims[K, D[2]]]) -> Array[Dims[K]]:
    K = velocities.shape[0]
    speed = np.linalg.norm(velocities, axis=1)
    heading = np.where(
        speed > 1e-6, np.arctan2(velocities[:, 1], velocities[:, 0]), 0.0
    )

    assert shape_of(heading, matches=(K,))

    return heading
