from typing import Sequence, cast
from dataclasses import dataclass

from trajax.type import DataType, jaxtyped
from trajax.mppi import JaxStateBatch
from trajax.costs.accelerated import JaxPositionExtractor, JaxHeadingExtractor
from trajax.costs.collision import (
    D_o,
    JaxObstacleStates,
    JaxSampledObstacleStates,
    JaxDistance,
)
from trajax.costs.distance.common import Circles

from numtypes import Array, Dims, D
from jaxtyping import Array as JaxArray, Float

import jax
import jax.numpy as jnp
import numpy as np


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxSampledObstaclePositionsAndHeading[T: int, K: int, N: int]:
    _x: Float[JaxArray, "T K N"]
    _y: Float[JaxArray, "T K N"]
    _heading: Float[JaxArray, "T K N"]

    @staticmethod
    def create[T_: int, K_: int, N_: int](
        *,
        x: Float[JaxArray, "T K N"],
        y: Float[JaxArray, "T K N"],
        heading: Float[JaxArray, "T K N"],
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
        sample_count: N_ | None = None,
    ) -> "JaxSampledObstaclePositionsAndHeading[T_, K_, N_]":
        return JaxSampledObstaclePositionsAndHeading(_x=x, _y=y, _heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        return np.stack([self._x, self._y, self._heading], axis=1)

    def x(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._x)

    def y(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._y)

    def heading(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._heading)

    @property
    def x_array(self) -> Float[JaxArray, "T K N"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "T K N"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "T K N"]:
        return self._heading

    @property
    def sample_count(self) -> N:
        return cast(N, self._x.shape[2])


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstaclePositionsAndHeading[T: int, K: int](JaxObstacleStates[T, K]):
    _x: Float[JaxArray, "T K"]
    _y: Float[JaxArray, "T K"]
    _heading: Float[JaxArray, "T K"]

    @staticmethod
    def sampled[N: int](  # type: ignore
        *,
        x: Float[JaxArray, "T K N"],
        y: Float[JaxArray, "T K N"],
        heading: Float[JaxArray, "T K N"],
        sample_count: N | None = None,
    ) -> JaxSampledObstaclePositionsAndHeading[T, K, N]:
        return JaxSampledObstaclePositionsAndHeading.create(
            x=x, y=y, heading=heading, sample_count=sample_count
        )

    @staticmethod
    def create[T_: int, K_: int](
        *,
        x: Float[JaxArray, "T K"],
        y: Float[JaxArray, "T K"],
        heading: Float[JaxArray, "T K"],
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
    ) -> "JaxObstaclePositionsAndHeading[T_, K_]":
        return JaxObstaclePositionsAndHeading(_x=x, _y=y, _heading=heading)

    @staticmethod
    def of_states[T_: int, K_: int](
        obstacle_states: Sequence[JaxObstacleStates[int, K_]],
        *,
        horizon: T_ | None = None,
    ) -> "JaxObstaclePositionsAndHeading[T_, K_]":
        assert horizon is None or len(obstacle_states) == horizon

        x = jnp.stack([states.x_array[0] for states in obstacle_states], axis=0)
        y = jnp.stack([states.y_array[0] for states in obstacle_states], axis=0)
        heading = jnp.stack(
            [states.heading_array[0] for states in obstacle_states], axis=0
        )

        return JaxObstaclePositionsAndHeading.create(
            x=x, y=y, heading=heading, horizon=horizon
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=-1)

    def x(self) -> Array[Dims[T, K]]:
        return np.asarray(self._x)

    def y(self) -> Array[Dims[T, K]]:
        return np.asarray(self._y)

    def heading(self) -> Array[Dims[T, K]]:
        return np.asarray(self._heading)

    def covariance(self) -> None:
        return

    def single(self) -> JaxSampledObstaclePositionsAndHeading[T, K, D[1]]:
        return JaxSampledObstaclePositionsAndHeading.create(
            x=self._x[..., jnp.newaxis],
            y=self._y[..., jnp.newaxis],
            heading=self._heading[..., jnp.newaxis],
        )

    @property
    def x_array(self) -> Float[JaxArray, "T K"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "T K"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        return self._heading

    @property
    def covariance_array(self) -> None:
        return None


@dataclass(frozen=True)
class JaxCircleDistanceExtractor[StateT: JaxStateBatch, V: int, C: int]:
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
    def create[S: JaxStateBatch, V_: int, C_: int](
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
            T, M = ego_positions.x.shape
            V = self._ego_circle_count
            N = obstacle_states.sample_count
            return JaxDistance(jnp.full((T, V, M, N), jnp.inf))

        return JaxDistance(
            compute_circle_distances(
                ego_x=ego_positions.x,
                ego_y=ego_positions.y,
                ego_heading=ego_headings.theta,
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
    x: Float[JaxArray, "T *S"],
    y: Float[JaxArray, "T *S"],
    heading: Float[JaxArray, "T *S"],
    local_origins: Float[JaxArray, "L 2"],
) -> tuple[Float[JaxArray, "L T *S"], Float[JaxArray, "L T *S"]]:
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
