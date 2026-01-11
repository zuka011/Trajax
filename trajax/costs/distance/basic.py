from typing import overload
from dataclasses import dataclass

from trajax.types import (
    DistanceExtractor,
    NumPyHeadings,
    NumPyPositions,
    NumPyPositionExtractor,
    NumPyHeadingExtractor,
)
from trajax.obstacles import NumPySampledObstacleStates
from trajax.costs.collision import NumPyDistance
from trajax.costs.distance.common import Circles

from numtypes import Array, Dims, D, shape_of

import numpy as np


type OriginsArray[N: int = int] = Array[Dims[N, D[2]]]
type RadiiArray[N: int = int] = Array[Dims[N]]


@dataclass(frozen=True)
class NumPyCircleDistanceExtractor[StateT, V: int, C: int](
    DistanceExtractor[StateT, NumPySampledObstacleStates, NumPyDistance]
):
    """
    Computes the distances between parts of the ego robot and obstacles. Both the ego
    and the obstacles are represented as collections of circles.
    """

    ego: Circles[V]
    obstacle: Circles[C]
    positions_from: NumPyPositionExtractor[StateT]
    headings_from: NumPyHeadingExtractor[StateT]

    @staticmethod
    def create[S, V_: int, C_: int](
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

    def __call__[T: int, N: int](
        self,
        *,
        states: StateT,
        obstacle_states: NumPySampledObstacleStates[T, int, N],
    ) -> NumPyDistance[T, V, int, N]:
        return NumPyDistance(
            compute_circle_distances(
                ego_positions=self.positions_from(states),
                ego_headings=self.headings_from(states),
                ego=self.ego,
                obstacle_states=replace_missing(obstacle_states),
                obstacle=self.obstacle,
            )
        )


def replace_missing[T: int, K: int, N: int](
    obstacle_states: NumPySampledObstacleStates[T, K, N],
) -> NumPySampledObstacleStates[T, K, N]:
    return NumPySampledObstacleStates.create(
        x=np.nan_to_num(obstacle_states.x(), nan=np.inf),
        y=np.nan_to_num(obstacle_states.y(), nan=np.inf),
        heading=np.nan_to_num(obstacle_states.heading(), nan=0.0),
    )


def compute_circle_distances[T: int, M: int, V: int, C: int, K: int, N: int](
    *,
    ego_positions: NumPyPositions[T, M],
    ego_headings: NumPyHeadings[T, M],
    ego: Circles[V],
    obstacle_states: NumPySampledObstacleStates[T, K, N],
    obstacle: Circles[C],
) -> Array[Dims[T, V, M, N]]:
    ego_global_x, ego_global_y = to_global_positions(
        x=ego_positions.x(),
        y=ego_positions.y(),
        heading=ego_headings.heading,
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


@overload
def to_global_positions[T: int, M: int, V: int](
    *,
    x: Array[Dims[T, M]],
    y: Array[Dims[T, M]],
    heading: Array[Dims[T, M]],
    local_origins: OriginsArray[V],
) -> tuple[Array[Dims[V, T, M]], Array[Dims[V, T, M]]]: ...


@overload
def to_global_positions[T: int, K: int, N: int, C: int](
    *,
    x: Array[Dims[T, K, N]],
    y: Array[Dims[T, K, N]],
    heading: Array[Dims[T, K, N]],
    local_origins: OriginsArray[C],
) -> tuple[Array[Dims[C, T, K, N]], Array[Dims[C, T, K, N]]]: ...


def to_global_positions[T: int, M: int, K: int, N: int, V: int, C: int](
    *,
    x: Array[Dims[T, M]] | Array[Dims[T, K, N]],
    y: Array[Dims[T, M]] | Array[Dims[T, K, N]],
    heading: Array[Dims[T, M]] | Array[Dims[T, K, N]],
    local_origins: OriginsArray[V] | OriginsArray[C],
) -> (
    tuple[Array[Dims[V, T, M]], Array[Dims[V, T, M]]]
    | tuple[Array[Dims[C, T, K, N]], Array[Dims[C, T, K, N]]]
):
    local_xy = local_origins.reshape((-1, 2) + (1,) * x.ndim)
    local_x, local_y = local_xy[:, 0], local_xy[:, 1]

    cos_h, sin_h = np.cos(heading), np.sin(heading)

    return (
        x + local_x * cos_h - local_y * sin_h,
        y + local_x * sin_h + local_y * cos_h,
    )


def pairwise_min_distances[T: int, M: int, V: int, C: int, K: int, N: int](
    ego_x: Array[Dims[V, T, M]],
    ego_y: Array[Dims[V, T, M]],
    ego_radii: RadiiArray[V],
    obstacle_x: Array[Dims[C, T, K, N]],
    obstacle_y: Array[Dims[C, T, K, N]],
    obstacle_radii: RadiiArray[C],
) -> Array[Dims[V, C, T, M, K, N]]:
    dx = (
        ego_x[:, np.newaxis, :, :, np.newaxis, np.newaxis]
        - obstacle_x[np.newaxis, :, :, np.newaxis, :, :]
    )
    dy = (
        ego_y[:, np.newaxis, :, :, np.newaxis, np.newaxis]
        - obstacle_y[np.newaxis, :, :, np.newaxis, :, :]
    )

    center_distances = np.sqrt(dx**2 + dy**2)
    radii_sum = (
        ego_radii[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        + obstacle_radii[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    )

    return center_distances - radii_sum


def min_distance_per_ego_part[T: int, M: int, V: int, C: int, K: int, N: int](
    pairwise_distances: Array[Dims[V, C, T, M, K, N]],
) -> Array[Dims[T, V, M, N]]:
    V, C, T, M, K, N = pairwise_distances.shape

    if C == 0 or K == 0:
        distances = np.full((T, V, M, N), np.inf)

        assert shape_of(distances, matches=(T, V, M, N))

        return distances

    min_over_obstacles = np.min(pairwise_distances, axis=(1, 4))
    return np.transpose(min_over_obstacles, (1, 0, 2, 3))
