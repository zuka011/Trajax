from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    DistanceExtractor,
    JaxPositionExtractor,
    JaxHeadingExtractor,
)
from trajax.obstacles import JaxSampledObstacleStates
from trajax.costs.collision import JaxDistance
from trajax.costs.distance.common import Circles

from jaxtyping import Array as JaxArray, Float

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class JaxCircleDistanceExtractor[StateT, V: int, C: int](
    DistanceExtractor[StateT, JaxSampledObstacleStates, JaxDistance]
):
    """
    Computes the distances between parts of the ego robot and obstacles. Both the ego
    and the obstacles are represented as collections of circles.
    """

    ego_origins: Float[JaxArray, "V 2"]
    ego_radii: Float[JaxArray, "V"]
    obstacle_origins: Float[JaxArray, "C 2"]
    obstacle_radii: Float[JaxArray, "C"]
    positions_from: JaxPositionExtractor[StateT]
    headings_from: JaxHeadingExtractor[StateT]

    @staticmethod
    def create[S, V_: int, C_: int](
        *,
        ego: Circles[V_],
        obstacle: Circles[C_],
        position_extractor: JaxPositionExtractor[S],
        heading_extractor: JaxHeadingExtractor[S],
    ) -> "JaxCircleDistanceExtractor[S, V_, C_]":
        return JaxCircleDistanceExtractor(
            ego_origins=jnp.asarray(ego.origins),
            ego_radii=jnp.asarray(ego.radii),
            obstacle_origins=jnp.asarray(obstacle.origins),
            obstacle_radii=jnp.asarray(obstacle.radii),
            positions_from=position_extractor,
            headings_from=heading_extractor,
        )

    def __call__[T: int, N: int](
        self,
        *,
        states: StateT,
        obstacle_states: JaxSampledObstacleStates[T, int, N],
    ) -> JaxDistance[T, V, int, N]:
        ego_positions = self.positions_from(states)
        ego_headings = self.headings_from(states)

        if self._no_obstacles_exist(obstacle_states):
            T, M = ego_positions.horizon, ego_positions.rollout_count
            V = self._ego_circle_count
            N = obstacle_states.sample_count
            return JaxDistance(jnp.full((T, V, M, N), jnp.inf))

        return JaxDistance(
            compute_circle_distances(
                ego_x=ego_positions.x_array,
                ego_y=ego_positions.y_array,
                ego_heading=ego_headings.heading,
                ego_origins=self.ego_origins,
                ego_radii=self.ego_radii,
                obstacle_x=obstacle_states.x_array,
                obstacle_y=obstacle_states.y_array,
                obstacle_heading=obstacle_states.heading_array,
                obstacle_origins=self.obstacle_origins,
                obstacle_radii=self.obstacle_radii,
            )
        )

    def _no_obstacles_exist(self, states: JaxSampledObstacleStates) -> bool:
        return self._obstacle_circle_count == 0 or states.x_array.shape[1] == 0

    @property
    def _obstacle_circle_count(self) -> int:
        return self.obstacle_origins.shape[0]

    @property
    def _ego_circle_count(self) -> int:
        return self.ego_origins.shape[0]


@jax.jit
@jaxtyped
def compute_circle_distances(
    *,
    ego_x: Float[JaxArray, "T M"],
    ego_y: Float[JaxArray, "T M"],
    ego_heading: Float[JaxArray, "T M"],
    ego_origins: Float[JaxArray, "V 2"],
    ego_radii: Float[JaxArray, "V"],
    obstacle_x: Float[JaxArray, "T K N"],
    obstacle_y: Float[JaxArray, "T K N"],
    obstacle_heading: Float[JaxArray, "T K N"],
    obstacle_origins: Float[JaxArray, "C 2"],
    obstacle_radii: Float[JaxArray, "C"],
) -> Float[JaxArray, "T V M N"]:
    ego_global_x, ego_global_y = to_global_positions(
        x=ego_x, y=ego_y, heading=ego_heading, local_origins=ego_origins
    )

    obstacle_global_x, obstacle_global_y = to_global_positions(
        x=obstacle_x,
        y=obstacle_y,
        heading=obstacle_heading,
        local_origins=obstacle_origins,
    )

    pairwise = pairwise_distances(
        ego_x=ego_global_x,
        ego_y=ego_global_y,
        ego_radii=ego_radii,
        obstacle_x=obstacle_global_x,
        obstacle_y=obstacle_global_y,
        obstacle_radii=obstacle_radii,
    )

    return jnp.min(pairwise, axis=(1, 4)).transpose((1, 0, 2, 3))


@jax.jit
@jaxtyped
def to_global_positions(
    *,
    x: Float[JaxArray, "T *S"],
    y: Float[JaxArray, "T *S"],
    heading: Float[JaxArray, "T *S"],
    local_origins: Float[JaxArray, "L 2"],
) -> tuple[Float[JaxArray, "L T *S"], Float[JaxArray, "L T *S"]]:
    x, y, heading = replace_missing(x=x, y=y, heading=heading)

    shape = (-1,) + (1,) * x.ndim
    local_x = local_origins[:, 0].reshape(shape)
    local_y = local_origins[:, 1].reshape(shape)

    cos_h, sin_h = jnp.cos(heading), jnp.sin(heading)

    return (
        x + local_x * cos_h - local_y * sin_h,
        y + local_x * sin_h + local_y * cos_h,
    )


@jax.jit
@jaxtyped
def replace_missing(
    *,
    x: Float[JaxArray, "*S"],
    y: Float[JaxArray, "*S"],
    heading: Float[JaxArray, "*S"],
) -> tuple[Float[JaxArray, "*S"], Float[JaxArray, "*S"], Float[JaxArray, "*S"]]:
    return (
        jnp.nan_to_num(x, nan=jnp.inf),
        jnp.nan_to_num(y, nan=jnp.inf),
        jnp.nan_to_num(heading, nan=0.0),
    )


@jax.jit
@jaxtyped
def pairwise_distances(
    *,
    ego_x: Float[JaxArray, "V T M"],
    ego_y: Float[JaxArray, "V T M"],
    ego_radii: Float[JaxArray, "V"],
    obstacle_x: Float[JaxArray, "C T K N"],
    obstacle_y: Float[JaxArray, "C T K N"],
    obstacle_radii: Float[JaxArray, "C"],
) -> Float[JaxArray, "V C T M K N"]:
    dx = ego_x[:, None, :, :, None, None] - obstacle_x[None, :, :, None, :, :]
    dy = ego_y[:, None, :, :, None, None] - obstacle_y[None, :, :, None, :, :]

    center_dist = jnp.sqrt(dx**2 + dy**2)
    radii_sum = (
        ego_radii[:, None, None, None, None, None]
        + obstacle_radii[None, :, None, None, None, None]
    )

    return center_dist - radii_sum
