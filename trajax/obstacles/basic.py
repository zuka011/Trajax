from dataclasses import dataclass

from trajax.costs import NumPyObstacleStateProvider, NumPyObstaclePositionsAndHeading

from numtypes import Array, Dims, D, shape_of

import numpy as np


@dataclass(frozen=True)
class NumPyStaticObstacleStateProvider[T: int, K: int](
    NumPyObstacleStateProvider[NumPyObstaclePositionsAndHeading[T, K]]
):
    positions: NumPyObstaclePositionsAndHeading[T, K]

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
            NumPyObstaclePositionsAndHeading.create(x=x, y=y, heading=heading)
        )

    def __call__(self) -> NumPyObstaclePositionsAndHeading[T, K]:
        return self.positions
