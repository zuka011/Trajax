from typing import Self, Any, cast
from dataclasses import dataclass

from trajax.types import NumPyObstacleStateProvider
from trajax.predictors import StaticPredictor
from trajax.obstacles.history import NumPyObstacleIds, NumPyObstacleStatesRunningHistory
from trajax.obstacles.basic import NumPyObstacleStates, NumPyObstacleStatesForTimeStep
from trajax.obstacles.common import PredictingObstacleStateProvider

from numtypes import Array, Dims, D, shape_of

import numpy as np


@dataclass(frozen=True)
class NumPyStaticObstacleStateProvider[T: int, K: int](
    NumPyObstacleStateProvider[NumPyObstacleStates[T, K]]
):
    inner: PredictingObstacleStateProvider[
        NumPyObstacleStatesForTimeStep[K],
        NumPyObstacleIds[K],
        NumPyObstacleStates[int, K],
        NumPyObstacleStates[T, K],
    ]

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
        count = positions.shape[0]
        headings = (
            headings
            if headings is not None
            else cast(Array[Dims[K_]], np.zeros(shape=(count,)))
        )

        return NumPyStaticObstacleStateProvider(  # type: ignore
            inner=PredictingObstacleStateProvider.create(
                predictor=StaticPredictor.create(horizon=horizon),
                history=NumPyObstacleStatesRunningHistory.single(
                    NumPyObstacleStatesForTimeStep.create(
                        x=positions[:, 0], y=positions[:, 1], heading=headings
                    ),
                ),
            ),
        )

    def with_time_step(self, time_step: float) -> Self:
        # Time step does not matter.
        return self

    def with_predictor(self, predictor: Any) -> Self:
        # Predictor does not matter.
        return self

    def __call__(self) -> NumPyObstacleStates[T, K]:
        return self.inner()

    def step(self) -> None:
        # Nothing to do, since the obstacles are static.
        pass
