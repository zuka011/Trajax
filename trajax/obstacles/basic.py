from dataclasses import dataclass

from trajax.costs import NumPyObstacleStateProvider, NumPyObstaclePositions

from numtypes import Array, Dims, D, shape_of

import numpy as np


@dataclass(frozen=True)
class NumPyStaticObstacleStateProvider[T: int, K: int](
    NumPyObstacleStateProvider[T, D[2], K]
):
    positions: NumPyObstaclePositions[T, K]

    @staticmethod
    def empty[T_: int](*, horizon: T_) -> "NumPyStaticObstacleStateProvider[T_, D[0]]":
        positions = np.empty((0, 2))

        assert shape_of(positions, matches=(0, 2))

        return NumPyStaticObstacleStateProvider.create(
            positions=positions, horizon=horizon
        )

    @staticmethod
    def create[T_: int, K_: int](
        positions: Array[Dims[K_, D[2]]], *, horizon: T_
    ) -> "NumPyStaticObstacleStateProvider[T_, K_]":
        K = positions.shape[0]
        x = np.tile(positions[:, 0], (horizon, 1))
        y = np.tile(positions[:, 1], (horizon, 1))

        assert shape_of(x, matches=(horizon, K))
        assert shape_of(y, matches=(horizon, K))

        return NumPyStaticObstacleStateProvider(NumPyObstaclePositions.create(x=x, y=y))

    def __call__(self) -> NumPyObstaclePositions[T, K]:
        return self.positions
