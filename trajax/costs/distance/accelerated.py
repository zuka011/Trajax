from typing import Protocol, Sequence
from dataclasses import dataclass

from trajax.type import DataType, jaxtyped
from trajax.mppi import JaxStateBatch
from trajax.costs.accelerated import JaxPositionExtractor, JaxDistance
from trajax.costs.common import ObstacleStateProvider
from trajax.costs.distance.common import Circles

from numtypes import Array, Dims, D
from jaxtyping import Array as JaxArray, Float

import jax
import jax.numpy as jnp
import numpy as np


class JaxObstacleStates[T: int, D_o: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        """Returns the states of obstacles as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, K]]:
        """Returns the x positions of obstacles over time as a NumPy array."""
        ...

    def y(self) -> Array[Dims[T, K]]:
        """Returns the y positions of obstacles over time as a NumPy array."""
        ...

    def heading(self) -> Array[Dims[T, K]]:
        """Returns the headings of obstacles over time as a NumPy array."""
        ...

    @property
    def x_array(self) -> Float[JaxArray, "T K"]:
        """Returns the x positions of obstacles over time."""
        ...

    @property
    def y_array(self) -> Float[JaxArray, "T K"]:
        """Returns the y positions of obstacles over time."""
        ...

    @property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        """Returns the headings of obstacles over time."""
        ...


class JaxObstacleStateProvider[T: int, D_o: int, K: int](
    ObstacleStateProvider[JaxObstacleStates[T, D_o, K]]
): ...


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstaclePositions[T: int, K: int](JaxObstacleStates[T, D[3], K]):
    _x: Float[JaxArray, "T K"]
    _y: Float[JaxArray, "T K"]
    _heading: Float[JaxArray, "T K"]

    @staticmethod
    def create[T_: int, K_: int](
        *,
        x: Float[JaxArray, "T K"],
        y: Float[JaxArray, "T K"],
        heading: Float[JaxArray, "T K"] | None = None,
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
    ) -> "JaxObstaclePositions[T_, K_]":
        assert horizon is None or horizon == x.shape[0] == y.shape[0], (
            f"Expected horizon {horizon}, but got x with shape {x.shape} and y with shape {y.shape}."
        )
        assert obstacle_count is None or obstacle_count == x.shape[1] == y.shape[1], (
            f"Expected obstacle count {obstacle_count}, but got x with shape {x.shape} and y with shape {y.shape}."
        )

        if heading is None:
            heading = jnp.zeros_like(x)

        return JaxObstaclePositions(_x=x, _y=y, _heading=heading)

    @staticmethod
    def of_states[T_: int, D_o_: int, K_: int](
        obstacle_states: Sequence[JaxObstacleStates[int, D_o_, K_]],
        *,
        horizon: T_ | None = None,
    ) -> "JaxObstaclePositions[T_, K_]":
        x = jnp.stack([states.x_array[0] for states in obstacle_states], axis=0)
        y = jnp.stack([states.y_array[0] for states in obstacle_states], axis=0)
        heading = jnp.stack(
            [states.heading_array[0] for states in obstacle_states], axis=0
        )

        return JaxObstaclePositions.create(x=x, y=y, heading=heading, horizon=horizon)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[3], K]]:
        return np.stack([self._x, self._y, self._heading], axis=-1)

    def x(self) -> Array[Dims[T, K]]:
        return np.asarray(self._x)

    def y(self) -> Array[Dims[T, K]]:
        return np.asarray(self._y)

    def heading(self) -> Array[Dims[T, K]]:
        return np.asarray(self._heading)

    @property
    def x_array(self) -> Float[JaxArray, "T K"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "T K"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        return self._heading


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
    obstacle_states: JaxObstacleStateProvider

    @staticmethod
    def create[S: JaxStateBatch, V_: int, C_: int](
        *,
        ego: Circles[V_],
        obstacle: Circles[C_],
        position_extractor: JaxPositionExtractor[S],
        obstacle_states: JaxObstacleStateProvider,
    ) -> "JaxCircleDistanceExtractor[S, V_, C_]":
        return JaxCircleDistanceExtractor(
            ego_origins=jnp.asarray(ego.origins),
            ego_radii=jnp.asarray(ego.radii),
            obstacle_origins=jnp.asarray(obstacle.origins),
            obstacle_radii=jnp.asarray(obstacle.radii),
            positions_from=position_extractor,
            obstacle_states=obstacle_states,
        )

    def __call__(self, states: StateT) -> JaxDistance[int, V, int]:
        return self.measure(states=states, obstacle_states=self.obstacle_states())

    def measure(
        self, *, states: StateT, obstacle_states: JaxObstacleStates
    ) -> JaxDistance[int, V, int]:
        ego_positions = self.positions_from(states)

        if self._no_obstacles_exist(obstacle_states):
            (T, M), V = ego_positions.x.shape, self._ego_circle_count
            return JaxDistance(jnp.full((T, V, M), jnp.inf))

        return JaxDistance(
            compute_circle_distances(
                ego_x=ego_positions.x,
                ego_y=ego_positions.y,
                ego_radii=self.ego_radii,
                ego_origins=self.ego_origins,
                obstacle_x=obstacle_states.x_array,
                obstacle_y=obstacle_states.y_array,
                obstacle_heading=obstacle_states.heading_array,
                obstacle_radii=self.obstacle_radii,
                obstacle_origins=self.obstacle_origins,
            )
        )

    def _no_obstacles_exist(self, states: JaxObstacleStates) -> bool:
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
    ego_origins: Float[JaxArray, "V 2"],
    ego_radii: Float[JaxArray, "V"],
    obstacle_x: Float[JaxArray, "T K"],
    obstacle_y: Float[JaxArray, "T K"],
    obstacle_heading: Float[JaxArray, "T K"],
    obstacle_origins: Float[JaxArray, "C 2"],
    obstacle_radii: Float[JaxArray, "C"],
) -> Float[JaxArray, "T V M"]:
    ego_global_x, ego_global_y = to_global_positions(
        x=ego_x, y=ego_y, local_origins=ego_origins
    )

    obstacle_global_x, obstacle_global_y = to_global_obstacle_positions(
        x=obstacle_x,
        y=obstacle_y,
        heading=obstacle_heading,
        local_origins=obstacle_origins,
    )

    pairwise_distances = pairwise_min_distances(
        ego_x=ego_global_x,
        ego_y=ego_global_y,
        ego_radii=ego_radii,
        obstacle_x=obstacle_global_x,
        obstacle_y=obstacle_global_y,
        obstacle_radii=obstacle_radii,
    )

    return min_distance_per_ego_part(pairwise_distances)


@jax.jit
@jaxtyped
def to_global_positions(
    *,
    x: Float[JaxArray, "A B"],
    y: Float[JaxArray, "A B"],
    local_origins: Float[JaxArray, "C 2"],
) -> tuple[Float[JaxArray, "C A B"], Float[JaxArray, "C A B"]]:
    global_x = x[jnp.newaxis, :, :] + local_origins[:, 0:1, jnp.newaxis]
    global_y = y[jnp.newaxis, :, :] + local_origins[:, 1:2, jnp.newaxis]

    return global_x, global_y


@jax.jit
@jaxtyped
def to_global_obstacle_positions(
    *,
    x: Float[JaxArray, "T K"],
    y: Float[JaxArray, "T K"],
    heading: Float[JaxArray, "T K"],
    local_origins: Float[JaxArray, "C 2"],
) -> tuple[Float[JaxArray, "C T K"], Float[JaxArray, "C T K"]]:
    """Computes global positions of obstacle circles considering heading rotation."""
    cos_h = jnp.cos(heading)
    sin_h = jnp.sin(heading)

    local_x = local_origins[:, 0:1, jnp.newaxis]
    local_y = local_origins[:, 1:2, jnp.newaxis]

    rotated_local_x = (
        local_x * cos_h[jnp.newaxis, :, :] - local_y * sin_h[jnp.newaxis, :, :]
    )
    rotated_local_y = (
        local_x * sin_h[jnp.newaxis, :, :] + local_y * cos_h[jnp.newaxis, :, :]
    )

    global_x = x[jnp.newaxis, :, :] + rotated_local_x
    global_y = y[jnp.newaxis, :, :] + rotated_local_y

    return global_x, global_y


@jax.jit
@jaxtyped
def pairwise_min_distances(
    *,
    ego_x: Float[JaxArray, "V T M"],
    ego_y: Float[JaxArray, "V T M"],
    ego_radii: Float[JaxArray, "V"],
    obstacle_x: Float[JaxArray, "C T K"],
    obstacle_y: Float[JaxArray, "C T K"],
    obstacle_radii: Float[JaxArray, "C"],
) -> Float[JaxArray, "V C T M K"]:
    dx = (
        ego_x[:, jnp.newaxis, :, :, jnp.newaxis]
        - obstacle_x[jnp.newaxis, :, :, jnp.newaxis, :]
    )
    dy = (
        ego_y[:, jnp.newaxis, :, :, jnp.newaxis]
        - obstacle_y[jnp.newaxis, :, :, jnp.newaxis, :]
    )

    center_distances = jnp.sqrt(dx**2 + dy**2)
    radii_sum = (
        ego_radii[:, jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        + obstacle_radii[jnp.newaxis, :, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    )

    return center_distances - radii_sum


@jax.jit
@jaxtyped
def min_distance_per_ego_part(
    pairwise_distances: Float[JaxArray, "V C T M K"],
) -> Float[JaxArray, "T V M"]:
    min_over_obstacles = jnp.min(pairwise_distances, axis=(1, 4))
    return jnp.transpose(min_over_obstacles, (1, 0, 2))
