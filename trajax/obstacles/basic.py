from typing import Sequence
from dataclasses import dataclass

from trajax.types import (
    DataType,
    D_o,
    D_O,
    NumPySampledObstacleStates,
    NumPyObstacleStates,
    NumPyObstacleStateProvider,
)

from numtypes import Array, Dims, D, shape_of

import numpy as np


type ObstacleCovarianceArray[T: int = int, K: int = int] = Array[Dims[T, D_o, D_o, K]]


@dataclass(kw_only=True, frozen=True)
class NumPySampledObstaclePositionsAndHeadings[T: int, K: int, N: int](
    NumPySampledObstacleStates[T, K, N]
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
    ) -> "NumPySampledObstaclePositionsAndHeadings[T_, K_, N_]":
        return NumPySampledObstaclePositionsAndHeadings(_x=x, _y=y, _heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        return np.stack([self._x, self._y, self._heading], axis=1)

    def x(self) -> Array[Dims[T, K, N]]:
        return self._x

    def y(self) -> Array[Dims[T, K, N]]:
        return self._y

    def heading(self) -> Array[Dims[T, K, N]]:
        return self._heading


@dataclass(kw_only=True, frozen=True)
class NumPyObstaclePositionsAndHeadings[T: int, K: int](NumPyObstacleStates[T, K]):
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
    ) -> NumPySampledObstaclePositionsAndHeadings[T_, K_, N_]:
        return NumPySampledObstaclePositionsAndHeadings.create(
            x=x, y=y, heading=heading
        )

    @staticmethod
    def create[T_: int, K_: int](
        *,
        x: Array[Dims[T_, K_]],
        y: Array[Dims[T_, K_]],
        heading: Array[Dims[T_, K_]],
        covariance: ObstacleCovarianceArray[T_, K_] | None = None,
    ) -> "NumPyObstaclePositionsAndHeadings[T_, K_]":
        return NumPyObstaclePositionsAndHeadings(
            _x=x, _y=y, _heading=heading, _covariance=covariance
        )

    @staticmethod
    def of_states[T_: int, K_: int](
        obstacle_states: Sequence["NumPyObstaclePositionsAndHeadings[int, K_]"],
        *,
        horizon: T_ | None = None,
    ) -> "NumPyObstaclePositionsAndHeadings[T_, K_]":
        assert horizon is None or len(obstacle_states) == horizon, (
            f"Expected horizon {horizon}, but got {len(obstacle_states)} obstacle states."
        )

        x = np.stack([states.x()[0] for states in obstacle_states], axis=0)
        y = np.stack([states.y()[0] for states in obstacle_states], axis=0)
        heading = np.stack([states.heading()[0] for states in obstacle_states], axis=0)

        return NumPyObstaclePositionsAndHeadings.create(x=x, y=y, heading=heading)

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

    def single(self) -> NumPySampledObstaclePositionsAndHeadings[T, K, D[1]]:
        return NumPySampledObstaclePositionsAndHeadings.create(
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


@dataclass(frozen=True)
class NumPyStaticObstacleStateProvider[T: int, K: int](
    NumPyObstacleStateProvider[NumPyObstaclePositionsAndHeadings[T, K]]
):
    positions: NumPyObstaclePositionsAndHeadings[T, K]

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
            NumPyObstaclePositionsAndHeadings.create(x=x, y=y, heading=heading)
        )

    def __call__(self) -> NumPyObstaclePositionsAndHeadings[T, K]:
        return self.positions
