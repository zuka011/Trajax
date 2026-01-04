from typing import Self, Any
from dataclasses import dataclass

from trajax.types import NumPyObstacleStateProvider
from trajax.obstacles.basic import NumPyObstacleStates

from numtypes import Array, Dims, D, shape_of

import numpy as np


@dataclass(frozen=True)
class NumPyStaticObstacleStateProvider[T: int, K: int](
    NumPyObstacleStateProvider[NumPyObstacleStates[T, K]]
):
    states: NumPyObstacleStates[T, K]

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
        return self.states

    def step(self) -> None:
        # Nothing to do, since the obstacles are static.
        pass
