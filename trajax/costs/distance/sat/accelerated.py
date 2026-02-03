from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    DistanceExtractor,
    JaxPositionExtractor,
    JaxHeadingExtractor,
    JaxSampledObstaclePositions,
    JaxSampledObstaclePositionExtractor,
    JaxSampledObstacleHeadingExtractor,
)
from trajax.costs.collision import JaxDistance
from trajax.costs.distance.sat.common import ConvexPolygon
from trajax.costs.distance.accelerated import replace_missing

from jaxtyping import Array as JaxArray, Float, Scalar, Bool
from numtypes import D

import jax
import jax.numpy as jnp

type V = D[1]
"""Number of ego polygons (fixed to 1 for now)."""


type BoolScalar = Bool[JaxArray, ""]
type Vector2d = Float[JaxArray, "2"]


@dataclass(frozen=True)
class JaxSatDistanceExtractor[StateT, SampledObstacleStatesT](
    DistanceExtractor[StateT, SampledObstacleStatesT, JaxDistance]
):
    """
    Computes the signed distances between the ego polygon and obstacle polygons
    using the Separating Axis Theorem (SAT).

    Positive distance means separated, negative means penetrating.
    """

    ego_vertices: Float[JaxArray, "P_ego 2"]
    obstacle_vertices: Float[JaxArray, "P_obs 2"]
    positions_from: JaxPositionExtractor[StateT]
    headings_from: JaxHeadingExtractor[StateT]
    obstacle_positions_from: JaxSampledObstaclePositionExtractor[SampledObstacleStatesT]
    obstacle_headings_from: JaxSampledObstacleHeadingExtractor[SampledObstacleStatesT]

    @staticmethod
    def create[S, SOS](
        *,
        ego: ConvexPolygon,
        obstacle: ConvexPolygon,
        position_extractor: JaxPositionExtractor[S],
        heading_extractor: JaxHeadingExtractor[S],
        obstacle_position_extractor: JaxSampledObstaclePositionExtractor[SOS],
        obstacle_heading_extractor: JaxSampledObstacleHeadingExtractor[SOS],
    ) -> "JaxSatDistanceExtractor[S, SOS]":
        return JaxSatDistanceExtractor(
            ego_vertices=jnp.asarray(ego.vertices),
            obstacle_vertices=jnp.asarray(obstacle.vertices),
            positions_from=position_extractor,
            headings_from=heading_extractor,
            obstacle_positions_from=obstacle_position_extractor,
            obstacle_headings_from=obstacle_heading_extractor,
        )

    def __call__[T: int = int, N: int = int, M: int = int](
        self, *, states: StateT, obstacle_states: SampledObstacleStatesT
    ) -> JaxDistance[T, V, M, N]:
        ego_positions = self.positions_from(states)
        ego_headings = self.headings_from(states)
        obstacle_positions = self.obstacle_positions_from(obstacle_states)
        obstacle_headings = self.obstacle_headings_from(obstacle_states)

        if self._no_obstacles_exist(obstacle_positions):
            T = ego_positions.horizon
            M = ego_positions.rollout_count
            V = 1
            N = obstacle_positions.sample_count
            return JaxDistance(jnp.full((T, V, M, N), jnp.inf))

        return JaxDistance(
            compute_sat_distances(
                ego_x=ego_positions.x_array,
                ego_y=ego_positions.y_array,
                ego_heading=ego_headings.heading_array,
                ego_vertices=self.ego_vertices,
                obstacle_x=obstacle_positions.x_array,
                obstacle_y=obstacle_positions.y_array,
                obstacle_heading=obstacle_headings.heading_array,
                obstacle_vertices=self.obstacle_vertices,
            )
        )

    def _no_obstacles_exist(self, positions: JaxSampledObstaclePositions) -> bool:
        return positions.x_array.shape[1] == 0


@jax.jit
@jaxtyped
def compute_sat_distances(
    *,
    ego_x: Float[JaxArray, "T M"],
    ego_y: Float[JaxArray, "T M"],
    ego_heading: Float[JaxArray, "T M"],
    ego_vertices: Float[JaxArray, "P_ego 2"],
    obstacle_x: Float[JaxArray, "T K N"],
    obstacle_y: Float[JaxArray, "T K N"],
    obstacle_heading: Float[JaxArray, "T K N"],
    obstacle_vertices: Float[JaxArray, "P_obs 2"],
) -> Float[JaxArray, "T V M N"]:
    obstacle_x, obstacle_y, obstacle_heading = replace_missing(
        x=obstacle_x, y=obstacle_y, heading=obstacle_heading
    )

    def compute_single_distance(
        ego_x_single: Scalar,
        ego_y_single: Scalar,
        ego_heading_single: Scalar,
        obs_x_single: Scalar,
        obs_y_single: Scalar,
        obs_heading_single: Scalar,
    ) -> Scalar:
        is_inf = jnp.isinf(obs_x_single) | jnp.isinf(obs_y_single)

        ego_transformed = transform_polygon_jax(
            ego_vertices,
            position=jnp.array([ego_x_single, ego_y_single]),
            heading=ego_heading_single,
        )
        obs_transformed = transform_polygon_jax(
            obstacle_vertices,
            position=jnp.array([obs_x_single, obs_y_single]),
            heading=obs_heading_single,
        )

        dist = sat_distance_jax(ego_transformed, obs_transformed)
        return jnp.where(is_inf, jnp.inf, dist)  # type: ignore

    def compute_for_ego_and_sample(
        ego_x_tm: Scalar,
        ego_y_tm: Scalar,
        ego_heading_tm: Scalar,
        obs_x_kn: Float[JaxArray, "K"],
        obs_y_kn: Float[JaxArray, "K"],
        obs_heading_kn: Float[JaxArray, "K"],
    ) -> Scalar:
        distances_k = jax.vmap(
            lambda ox, oy, oh: compute_single_distance(
                ego_x_tm, ego_y_tm, ego_heading_tm, ox, oy, oh
            )
        )(obs_x_kn, obs_y_kn, obs_heading_kn)
        return jnp.min(distances_k)

    def compute_for_timestep(
        ego_x_t: Float[JaxArray, "M"],
        ego_y_t: Float[JaxArray, "M"],
        ego_heading_t: Float[JaxArray, "M"],
        obs_x_t: Float[JaxArray, "K N"],
        obs_y_t: Float[JaxArray, "K N"],
        obs_heading_t: Float[JaxArray, "K N"],
    ) -> Float[JaxArray, "M N"]:
        def compute_for_ego(ex, ey, eh):
            return jax.vmap(
                lambda ox_n, oy_n, oh_n: compute_for_ego_and_sample(
                    ex, ey, eh, ox_n, oy_n, oh_n
                ),
                in_axes=(1, 1, 1),
            )(obs_x_t, obs_y_t, obs_heading_t)

        return jax.vmap(compute_for_ego)(ego_x_t, ego_y_t, ego_heading_t)

    distances = jax.vmap(compute_for_timestep)(
        ego_x,
        ego_y,
        ego_heading,
        obstacle_x,
        obstacle_y,
        obstacle_heading,
    )

    return distances[:, None, :, :]


@jax.jit
@jaxtyped
def transform_polygon_jax(
    local_vertices: Float[JaxArray, "P 2"],
    position: Float[JaxArray, "2"],
    heading: Scalar,
) -> Float[JaxArray, "P 2"]:
    cos_h = jnp.cos(heading)
    sin_h = jnp.sin(heading)
    rotation = jnp.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return (local_vertices @ rotation.T) + position


@jax.jit
@jaxtyped
def get_edge_normals_jax(
    vertices: Float[JaxArray, "P 2"],
) -> Float[JaxArray, "P 2"]:
    edges = jnp.roll(vertices, -1, axis=0) - vertices
    normals = jnp.stack([-edges[:, 1], edges[:, 0]], axis=-1)
    norms = jnp.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / jnp.where(norms > 1e-10, norms, 1.0)  # type: ignore


@jax.jit
@jaxtyped
def project_polygon_jax(
    vertices: Float[JaxArray, "P 2"], axis: Vector2d
) -> tuple[Scalar, Scalar]:
    projections = vertices @ axis
    return projections.min(), projections.max()


@jax.jit
@jaxtyped
def sat_distance_jax(
    poly_a: Float[JaxArray, "P_a 2"],
    poly_b: Float[JaxArray, "P_b 2"],
) -> Scalar:
    normals_a = get_edge_normals_jax(poly_a)
    normals_b = get_edge_normals_jax(poly_b)
    axes = jnp.vstack([normals_a, normals_b])

    def check_axis(axis: Vector2d) -> tuple[BoolScalar, Scalar, Scalar]:
        min_a, max_a = project_polygon_jax(poly_a, axis)
        min_b, max_b = project_polygon_jax(poly_b, axis)

        gap_ab = min_b - max_a
        gap_ba = min_a - max_b
        gap = jnp.maximum(gap_ab, gap_ba)

        overlap = jnp.minimum(max_a - min_b, max_b - min_a)

        is_separated = gap > 0
        return is_separated, gap, -overlap

    is_separated_arr, gaps, penetrations = jax.vmap(check_axis)(axes)

    any_separated = jnp.any(is_separated_arr)
    min_separation = jnp.where(is_separated_arr, gaps, jnp.inf).min()
    max_penetration = jnp.where(~is_separated_arr, penetrations, -jnp.inf).max()

    return jnp.where(any_separated, min_separation, max_penetration)
