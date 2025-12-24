from typing import Sequence
from dataclasses import dataclass

from trajax.type import DataType
from trajax.mppi import NumPyStateBatch
from trajax.trajectory import NumPyHeadings
from trajax.costs.basic import (
    NumPyDistance,
    NumPyPositions,
    NumPyPositionExtractor,
    NumPyHeadingExtractor,
    NumPyObstacleStates,
)
from trajax.costs.distance.common import Circles

from numtypes import Array, Dims, D, shape_of

import numpy as np


type OriginsArray[N: int = int] = Array[Dims[N, D[2]]]
type RadiiArray[N: int = int] = Array[Dims[N]]


@dataclass(kw_only=True, frozen=True)
class NumPyObstaclePositions[T: int, K: int](NumPyObstacleStates[T, D[3], K]):
    _x: Array[Dims[T, K]]
    _y: Array[Dims[T, K]]
    _heading: Array[Dims[T, K]]

    @staticmethod
    def create[T_: int, K_: int](
        *,
        x: Array[Dims[T_, K_]],
        y: Array[Dims[T_, K_]],
        heading: Array[Dims[T_, K_]] | None = None,
    ) -> "NumPyObstaclePositions[T_, K_]":
        if heading is None:
            heading = np.zeros_like(x)

        return NumPyObstaclePositions(_x=x, _y=y, _heading=heading)

    @staticmethod
    def of_states[T_: int, D_o_: int, K_: int](
        obstacle_states: Sequence[NumPyObstacleStates[int, D_o_, K_]],
        *,
        horizon: T_ | None = None,
    ) -> "NumPyObstaclePositions[T_, K_]":
        x = np.stack([states.x()[0] for states in obstacle_states], axis=0)
        y = np.stack([states.y()[0] for states in obstacle_states], axis=0)
        heading = np.stack([states.heading()[0] for states in obstacle_states], axis=0)

        return NumPyObstaclePositions.create(x=x, y=y, heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[3], K]]:
        return np.stack([self._x, self._y, self._heading], axis=-1)

    def x(self) -> Array[Dims[T, K]]:
        return self._x

    def y(self) -> Array[Dims[T, K]]:
        return self._y

    def heading(self) -> Array[Dims[T, K]]:
        return self._heading


@dataclass(frozen=True)
class NumPyCircleDistanceExtractor[StateT: NumPyStateBatch, V: int, C: int]:
    """
    Computes the distances between parts of the ego robot and obstacles. Both the ego
    and the obstacles are represented as collections of circles.
    """

    ego: Circles[V]
    obstacle: Circles[C]
    positions_from: NumPyPositionExtractor[StateT]
    headings_from: NumPyHeadingExtractor[StateT]

    @staticmethod
    def create[S: NumPyStateBatch, V_: int, C_: int](
        *,
        ego: Circles[V_],
        obstacle: Circles[C_],
        position_extractor: NumPyPositionExtractor[S],
        heading_extractor: NumPyHeadingExtractor[S],
    ) -> "NumPyCircleDistanceExtractor[S, V_, C_]":
        return NumPyCircleDistanceExtractor(
            ego=ego,
            obstacle=obstacle,
            positions_from=position_extractor,
            headings_from=heading_extractor,
        )

    def __call__(
        self, *, states: StateT, obstacle_states: NumPyObstacleStates
    ) -> NumPyDistance[int, V, int]:
        return NumPyDistance(
            compute_circle_distances(
                ego_positions=self.positions_from(states),
                ego_headings=self.headings_from(states),
                ego=self.ego,
                obstacle_states=obstacle_states,
                obstacle=self.obstacle,
            )
        )


def compute_circle_distances[T: int, M: int, V: int, C: int, K: int](
    *,
    ego_positions: NumPyPositions[T, M],
    ego_headings: NumPyHeadings[T, M],
    ego: Circles[V],
    obstacle_states: NumPyObstacleStates[T, int, K],
    obstacle: Circles[C],
) -> Array[Dims[T, V, M]]:
    ego_global_x, ego_global_y = to_global_positions(
        x=ego_positions.x,
        y=ego_positions.y,
        heading=ego_headings.theta,
        local_origins=ego.origins,
    )

    obstacle_global_x, obstacle_global_y = to_global_positions(
        x=obstacle_states.x(),
        y=obstacle_states.y(),
        heading=obstacle_states.heading(),
        local_origins=obstacle.origins,
    )

    pairwise_distances = pairwise_min_distances(
        ego_x=ego_global_x,
        ego_y=ego_global_y,
        ego_radii=ego.radii,
        obstacle_x=obstacle_global_x,
        obstacle_y=obstacle_global_y,
        obstacle_radii=obstacle.radii,
    )

    return min_distance_per_ego_part(pairwise_distances)


def to_global_positions[A: int, B: int, C: int](
    x: Array[Dims[A, B]],
    y: Array[Dims[A, B]],
    heading: Array[Dims[A, B]],
    local_origins: OriginsArray[C],
) -> tuple[Array[Dims[C, A, B]], Array[Dims[C, A, B]]]:
    local_x = local_origins[:, 0:1, np.newaxis]
    local_y = local_origins[:, 1:2, np.newaxis]

    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    rotated_local_x = (
        local_x * cos_h[np.newaxis, :, :] - local_y * sin_h[np.newaxis, :, :]
    )
    rotated_local_y = (
        local_x * sin_h[np.newaxis, :, :] + local_y * cos_h[np.newaxis, :, :]
    )

    global_x = x[np.newaxis, :, :] + rotated_local_x
    global_y = y[np.newaxis, :, :] + rotated_local_y

    return global_x, global_y


def pairwise_min_distances[T: int, M: int, V: int, C: int, K: int](
    ego_x: Array[Dims[V, T, M]],
    ego_y: Array[Dims[V, T, M]],
    ego_radii: RadiiArray[V],
    obstacle_x: Array[Dims[C, T, K]],
    obstacle_y: Array[Dims[C, T, K]],
    obstacle_radii: RadiiArray[C],
) -> Array[Dims[V, C, T, M, K]]:
    dx = (
        ego_x[:, np.newaxis, :, :, np.newaxis]
        - obstacle_x[np.newaxis, :, :, np.newaxis, :]
    )
    dy = (
        ego_y[:, np.newaxis, :, :, np.newaxis]
        - obstacle_y[np.newaxis, :, :, np.newaxis, :]
    )

    center_distances = np.sqrt(dx**2 + dy**2)
    radii_sum = (
        ego_radii[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        + obstacle_radii[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    )

    return center_distances - radii_sum


def min_distance_per_ego_part[T: int, M: int, V: int, C: int, K: int](
    pairwise_distances: Array[Dims[V, C, T, M, K]],
) -> Array[Dims[T, V, M]]:
    V, C, T, M, K = pairwise_distances.shape

    if C == 0 or K == 0:
        distances = np.full((T, V, M), np.inf)
        assert shape_of(distances, matches=(T, V, M), name="distances")
        return distances

    min_over_obstacles = np.min(pairwise_distances, axis=(1, 4))
    return np.transpose(min_over_obstacles, (1, 0, 2))
