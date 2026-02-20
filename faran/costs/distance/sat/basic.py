from typing import NamedTuple, overload
from dataclasses import dataclass

from faran.types import (
    DistanceExtractor,
    NumPyHeadings,
    NumPyPositions,
    NumPyPositionExtractor,
    NumPyHeadingExtractor,
    NumPySampledObstaclePositions,
    NumPySampledObstacleHeadings,
    NumPySampledObstaclePositionExtractor,
    NumPySampledObstacleHeadingExtractor,
)
from faran.costs.collision import NumPyDistance
from faran.costs.distance.basic import replace_missing
from faran.costs.distance.sat.common import ConvexPolygon, VerticesArray

from numtypes import Array, BoolArray, Dims, D, shape_of

import numpy as np

type V = D[1]
"""Number of ego polygons (fixed to 1 for now)."""


class MaskedObstacleStates[T: int, K: int, N: int](NamedTuple):
    x: Array[Dims[T, K, N]]
    y: Array[Dims[T, K, N]]
    heading: Array[Dims[T, K, N]]
    is_valid: BoolArray[Dims[T, K, N]]


@dataclass(frozen=True)
class NumPySatDistanceExtractor[StateT, SampledObstacleStatesT](
    DistanceExtractor[StateT, SampledObstacleStatesT, NumPyDistance]
):
    """
    Computes the signed distances between the ego polygon and obstacle polygons
    using the Separating Axis Theorem (SAT).

    Positive distance means separated, negative means penetrating.
    """

    ego: ConvexPolygon
    obstacle: ConvexPolygon
    positions_from: NumPyPositionExtractor[StateT]
    headings_from: NumPyHeadingExtractor[StateT]
    obstacle_positions_from: NumPySampledObstaclePositionExtractor[
        SampledObstacleStatesT
    ]
    obstacle_headings_from: NumPySampledObstacleHeadingExtractor[SampledObstacleStatesT]

    @staticmethod
    def create[S, SOS](
        *,
        ego: ConvexPolygon,
        obstacle: ConvexPolygon,
        position_extractor: NumPyPositionExtractor[S],
        heading_extractor: NumPyHeadingExtractor[S],
        obstacle_position_extractor: NumPySampledObstaclePositionExtractor[SOS],
        obstacle_heading_extractor: NumPySampledObstacleHeadingExtractor[SOS],
    ) -> "NumPySatDistanceExtractor[S, SOS]":
        return NumPySatDistanceExtractor(
            ego=ego,
            obstacle=obstacle,
            positions_from=position_extractor,
            headings_from=heading_extractor,
            obstacle_positions_from=obstacle_position_extractor,
            obstacle_headings_from=obstacle_heading_extractor,
        )

    def __call__[T: int = int, N: int = int, M: int = int](
        self, *, states: StateT, obstacle_states: SampledObstacleStatesT
    ) -> NumPyDistance[T, V, M, N]:
        obstacle_positions, obstacle_headings = replace_missing(
            positions=self.obstacle_positions_from(obstacle_states),
            headings=self.obstacle_headings_from(obstacle_states),
        )

        return NumPyDistance(
            compute_sat_distances(
                ego_positions=self.positions_from(states),
                ego_headings=self.headings_from(states),
                ego_vertices=self.ego.vertices,
                obstacle_positions=obstacle_positions,
                obstacle_headings=obstacle_headings,
                obstacle_vertices=self.obstacle.vertices,
            )
        )


def compute_sat_distances[T: int, M: int, K: int, N: int](
    *,
    ego_positions: NumPyPositions[T, M],
    ego_headings: NumPyHeadings[T, M],
    ego_vertices: VerticesArray,
    obstacle_positions: NumPySampledObstaclePositions[T, K, N],
    obstacle_headings: NumPySampledObstacleHeadings[T, K, N],
    obstacle_vertices: VerticesArray,
) -> Array[Dims[T, V, M, N]]:
    ego_x = ego_positions.x()
    ego_y = ego_positions.y()
    ego_heading = ego_headings.heading()

    obstacle_x, obstacle_y, obstacle_heading, is_valid = mask_valid_obstacle_states(
        positions=obstacle_positions, headings=obstacle_headings
    )

    T, M = ego_x.shape
    K = obstacle_x.shape[1]
    N = obstacle_x.shape[2]
    V = 1

    if K == 0:
        distances = np.full((T, V, M, N), np.inf)
        assert shape_of(distances, matches=(T, V, M, N), name="distances")
        return distances

    ego_polygons = transform_polygons_batched(
        ego_vertices,
        x=ego_x,
        y=ego_y,
        heading=ego_heading,
    )

    obstacle_polygons = transform_polygons_batched(
        obstacle_vertices,
        x=obstacle_x,
        y=obstacle_y,
        heading=obstacle_heading,
    )

    distances = sat_distances_batched(
        ego_polygons=ego_polygons,
        obstacle_polygons=obstacle_polygons,
        is_valid=is_valid,
    )

    min_over_obstacles = np.min(distances, axis=2, keepdims=True)
    result = np.moveaxis(min_over_obstacles, 2, 1)

    return result


def mask_valid_obstacle_states[T: int, K: int, N: int](
    *,
    positions: NumPySampledObstaclePositions[T, K, N],
    headings: NumPySampledObstacleHeadings[T, K, N],
) -> MaskedObstacleStates[T, K, N]:
    x = positions.x()
    y = positions.y()
    heading = headings.heading()
    T, K, N = x.shape

    is_valid = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x = np.where(is_valid, x, 0.0)
    y = np.where(is_valid, y, 0.0)
    heading = np.where(is_valid, heading, 0.0)

    assert shape_of(x, matches=(T, K, N), name="x")
    assert shape_of(y, matches=(T, K, N), name="y")
    assert shape_of(heading, matches=(T, K, N), name="heading")
    assert shape_of(is_valid, matches=(T, K, N), name="is_valid")

    return MaskedObstacleStates(x=x, y=y, heading=heading, is_valid=is_valid)


@overload
def transform_polygons_batched[T: int, M: int, P: int = int](
    local_vertices: VerticesArray,
    *,
    x: Array[Dims[T, M]],
    y: Array[Dims[T, M]],
    heading: Array[Dims[T, M]],
) -> Array[Dims[T, M, P, D[2]]]: ...


@overload
def transform_polygons_batched[T: int, K: int, N: int, P: int = int](
    local_vertices: VerticesArray,
    *,
    x: Array[Dims[T, K, N]],
    y: Array[Dims[T, K, N]],
    heading: Array[Dims[T, K, N]],
) -> Array[Dims[T, K, N, P, D[2]]]: ...


def transform_polygons_batched[T: int, M: int, K: int, N: int, P: int = int](
    local_vertices: VerticesArray,
    *,
    x: Array[Dims[T, M]] | Array[Dims[T, K, N]],
    y: Array[Dims[T, M]] | Array[Dims[T, K, N]],
    heading: Array[Dims[T, M]] | Array[Dims[T, K, N]],
) -> Array[Dims[T, M, P, D[2]]] | Array[Dims[T, K, N, P, D[2]]]:
    P = local_vertices.shape[0]
    spatial_shape = x.shape

    rotation_matrices = create_rotation_matrices(heading)
    local_expanded = local_vertices.reshape((1,) * len(spatial_shape) + (P, 2))
    rotated = np.einsum("...pj,...ij->...pi", local_expanded, rotation_matrices)

    positions = np.stack([x, y], axis=-1)
    positions_expanded = positions[..., np.newaxis, :]

    return rotated + positions_expanded


@overload
def create_rotation_matrices[T: int, M: int](
    heading: Array[Dims[T, M]],
) -> Array[Dims[T, M, D[2], D[2]]]: ...


@overload
def create_rotation_matrices[T: int, K: int, N: int](
    heading: Array[Dims[T, K, N]],
) -> Array[Dims[T, K, N, D[2], D[2]]]: ...


def create_rotation_matrices[T: int, M: int, K: int, N: int](
    heading: Array[Dims[T, M]] | Array[Dims[T, K, N]],
) -> Array[Dims[T, M, D[2], D[2]]] | Array[Dims[T, K, N, D[2], D[2]]]:
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    return np.stack(
        [
            np.stack([cos_h, -sin_h], axis=-1),
            np.stack([sin_h, cos_h], axis=-1),
        ],
        axis=-2,
    )


def sat_distances_batched[
    T: int,
    M: int,
    K: int,
    N: int,
    P_ego: int = int,
    P_obs: int = int,
](
    *,
    ego_polygons: Array[Dims[T, M, P_ego, D[2]]],
    obstacle_polygons: Array[Dims[T, K, N, P_obs, D[2]]],
    is_valid: BoolArray[Dims[T, K, N]],
) -> Array[Dims[T, M, K, N]]:
    all_axes = gather_separating_axes(ego_polygons, obstacle_polygons)

    ego_expanded = ego_polygons[:, :, np.newaxis, np.newaxis, :, :]
    obs_expanded = obstacle_polygons[:, np.newaxis, :, :, :, :]

    ego_min, ego_max = project_polygons_onto_axes(ego_expanded, all_axes)
    obs_min, obs_max = project_polygons_onto_axes(obs_expanded, all_axes)

    distances = compute_signed_distances_from_projections(
        ego_min, ego_max, obs_min, obs_max
    )

    is_invalid_expanded = ~is_valid[:, np.newaxis, :, :]
    return np.where(is_invalid_expanded, np.inf, distances)


def gather_separating_axes[
    T: int,
    M: int,
    K: int,
    N: int,
    P_ego: int = int,
    P_obs: int = int,
    L_axes: int = int,
](
    ego_polygons: Array[Dims[T, M, P_ego, D[2]]],
    obstacle_polygons: Array[Dims[T, K, N, P_obs, D[2]]],
) -> Array[Dims[T, M, K, N, L_axes, D[2]]]:
    P_ego = ego_polygons.shape[2]
    P_obs = obstacle_polygons.shape[3]

    ego_normals = get_edge_normals_batched(ego_polygons)
    obs_normals = get_edge_normals_batched(obstacle_polygons)

    ego_normals_expanded = ego_normals[:, :, np.newaxis, np.newaxis, :, :]
    obs_normals_expanded = obs_normals[:, np.newaxis, :, :, :, :]

    T, M = ego_polygons.shape[:2]
    K, N = obstacle_polygons.shape[1:3]

    ego_normals_full = np.broadcast_to(ego_normals_expanded, (T, M, K, N, P_ego, 2))
    obs_normals_full = np.broadcast_to(obs_normals_expanded, (T, M, K, N, P_obs, 2))

    return np.concatenate([ego_normals_full, obs_normals_full], axis=-2)


@overload
def project_polygons_onto_axes[
    T: int,
    M: int,
    K: int,
    N: int,
    P: int = int,
    L_axes: int = int,
](
    polygons: Array[Dims[T, M, D[1], D[1], P, D[2]]],
    axes: Array[Dims[T, M, K, N, L_axes, D[2]]],
) -> tuple[Array[Dims[T, M, K, N, L_axes]], Array[Dims[T, M, K, N, L_axes]]]: ...


@overload
def project_polygons_onto_axes[
    T: int,
    M: int,
    K: int,
    N: int,
    P: int = int,
    L_axes: int = int,
](
    polygons: Array[Dims[T, D[1], K, N, P, D[2]]],
    axes: Array[Dims[T, M, K, N, L_axes, D[2]]],
) -> tuple[Array[Dims[T, M, K, N, L_axes]], Array[Dims[T, M, K, N, L_axes]]]: ...


def project_polygons_onto_axes[
    T: int,
    M: int,
    K: int,
    N: int,
    P: int = int,
    L_axes: int = int,
](
    polygons: Array[Dims[T, M, D[1], D[1], P, D[2]]]
    | Array[Dims[T, D[1], K, N, P, D[2]]],
    axes: Array[Dims[T, M, K, N, L_axes, D[2]]],
) -> tuple[Array[Dims[T, M, K, N, L_axes]], Array[Dims[T, M, K, N, L_axes]]]:
    projections = np.einsum("...pi,...ai->...ap", polygons, axes)
    return projections.min(axis=-1), projections.max(axis=-1)


def compute_signed_distances_from_projections[
    T: int,
    M: int,
    K: int,
    N: int,
    L_axes: int = int,
](
    ego_min: Array[Dims[T, M, K, N, L_axes]],
    ego_max: Array[Dims[T, M, K, N, L_axes]],
    obs_min: Array[Dims[T, M, K, N, L_axes]],
    obs_max: Array[Dims[T, M, K, N, L_axes]],
) -> Array[Dims[T, M, K, N]]:
    gap_ab = obs_min - ego_max
    gap_ba = ego_min - obs_max
    gaps = np.maximum(gap_ab, gap_ba)

    overlap = np.minimum(ego_max - obs_min, obs_max - ego_min)
    penetration = -overlap

    is_separated = gaps > 0
    any_separated = np.any(is_separated, axis=-1)

    separation_values = np.where(is_separated, gaps, np.inf)
    min_separation = np.min(separation_values, axis=-1)

    penetration_values = np.where(~is_separated, penetration, -np.inf)
    max_penetration = np.max(penetration_values, axis=-1)

    return np.where(any_separated, min_separation, max_penetration)


@overload
def get_edge_normals_batched[T: int, M: int, P_ego: int](
    vertices: Array[Dims[T, M, P_ego, D[2]]],
) -> Array[Dims[T, M, P_ego, D[2]]]: ...


@overload
def get_edge_normals_batched[T: int, K: int, N: int, P_obs: int](
    vertices: Array[Dims[T, K, N, P_obs, D[2]]],
) -> Array[Dims[T, K, N, P_obs, D[2]]]: ...


def get_edge_normals_batched[T: int, M: int, K: int, N: int, P: int](
    vertices: Array[Dims[T, M, P, D[2]]] | Array[Dims[T, K, N, P, D[2]]],
) -> Array[Dims[T, M, P, D[2]]] | Array[Dims[T, K, N, P, D[2]]]:
    edges = np.roll(vertices, -1, axis=-2) - vertices
    normals = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.where(norms > 1e-10, norms, 1.0)
