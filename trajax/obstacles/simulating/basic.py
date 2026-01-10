from typing import Self, Any, Final
from dataclasses import dataclass

from trajax.types import NumPyObstacleStateProvider, ObstacleMotionPredictor
from trajax.obstacles.assignment import NumPyHungarianObstacleIdAssignment
from trajax.obstacles.history import NumPyObstacleIds, NumPyObstacleStatesRunningHistory
from trajax.obstacles.basic import (
    NumPyObstacleStates,
    NumPyObstacleStatesForTimeStep,
    NumPyObstacle2dPositions,
    NumPyObstacle2dPositionsForTimeStep,
)
from trajax.obstacles.common import PredictingObstacleStateProvider

from numtypes import Array, Dims, D, shape_of

import numpy as np


HISTORY_HORIZON: Final = 3


class ObstaclePositionExtractor:
    def of_states_for_time_step[K: int](
        self, states: NumPyObstacleStatesForTimeStep[K], /
    ) -> NumPyObstacle2dPositionsForTimeStep[K]:
        return states.positions()

    def of_states[K: int](
        self, states: NumPyObstacleStates[int, K], /
    ) -> NumPyObstacle2dPositions[int, K]:
        return states.positions()


@dataclass(kw_only=True)
class NumPyDynamicObstacleStateProvider[PredictionT, K: int](
    NumPyObstacleStateProvider[PredictionT]
):
    type MotionPredictor[P, K_: int] = ObstacleMotionPredictor[
        NumPyObstacleStates[int, K_], P
    ]

    history: NumPyObstacleStatesRunningHistory[int, K]
    velocities: Array[Dims[K, D[2]]]

    time_step: float | None = None
    inner: (
        PredictingObstacleStateProvider[
            NumPyObstacleStatesForTimeStep[K],
            NumPyObstacleIds[K],
            NumPyObstacleStates[int, K],
            PredictionT,
        ]
        | None
    ) = None

    @staticmethod
    def create[K_: int = int](
        *,
        positions: Array[Dims[K_, D[2]]],
        velocities: Array[Dims[K_, D[2]]],
    ) -> "NumPyDynamicObstacleStateProvider[Any, K_]":
        headings = headings_from(velocities)

        return NumPyDynamicObstacleStateProvider(
            history=NumPyObstacleStatesRunningHistory.single(
                NumPyObstacleStatesForTimeStep.create(
                    x=positions[:, 0], y=positions[:, 1], heading=headings
                ),
                horizon=HISTORY_HORIZON,
                obstacle_count=positions.shape[0],
            ),
            velocities=velocities,
        )

    def with_time_step(self, time_step: float) -> Self:
        return self.__class__(
            history=self.history,
            velocities=self.velocities,
            time_step=time_step,
            inner=self.inner,
        )

    def with_predictor[P](
        self, predictor: MotionPredictor[P, K]
    ) -> "NumPyDynamicObstacleStateProvider[P, K]":
        return NumPyDynamicObstacleStateProvider(
            history=self.history,
            velocities=self.velocities,
            time_step=self.time_step,
            inner=PredictingObstacleStateProvider.create(
                predictor=predictor,
                history=self.history,
                id_assignment=NumPyHungarianObstacleIdAssignment.create(
                    position_extractor=ObstaclePositionExtractor(), cutoff=1.0
                ),
            ),
        )

    def __call__(self) -> PredictionT:
        assert self.inner is not None, (
            "Motion predictor must be set to provide obstacle states."
        )

        return self.inner()

    def step(self) -> None:
        assert self.time_step is not None, (
            "Time step must be set to advance obstacle states."
        )
        assert self.inner is not None, (
            "Motion predictor must be set to advance obstacle states."
        )

        last = self.history.last()
        x, y = step_obstacles(
            x=last.x(), y=last.y(), velocities=self.velocities, time_step=self.time_step
        )

        self.history = self.inner.history = self.history.append(
            NumPyObstacleStatesForTimeStep.create(x=x, y=y, heading=last.heading())
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
