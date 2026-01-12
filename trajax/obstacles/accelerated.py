from typing import Sequence, cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    jaxtyped,
    DataType,
    Device,
    place,
    D_o,
    D_O,
    SampledObstacleStates,
    ObstacleStates,
)
from trajax.obstacles.basic import NumPyObstacleStatesForTimeStep

from numtypes import Array, Dims, D
from jaxtyping import Array as JaxArray, Float

import numpy as np
import jax.numpy as jnp


type ObstacleCovarianceArray[T: int = int, K: int = int] = Float[
    JaxArray, f"T {D_O} {D_O} K"
]


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxSampledObstacleStates[T: int, K: int, N: int](SampledObstacleStates[T, K, N]):
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
    ) -> "JaxSampledObstacleStates[T_, K_, N_]":
        return JaxSampledObstacleStates(_x=x, _y=y, _heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        return self._numpy_array

    def x(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._x)

    def y(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._y)

    def heading(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._heading)

    def at(self, *, time_step: int, sample: int) -> "JaxObstacleStatesForTimeStep[K]":
        return JaxObstacleStatesForTimeStep.create(
            x=self._x[time_step, :, sample],
            y=self._y[time_step, :, sample],
            heading=self._heading[time_step, :, sample],
        )

    @property
    def horizon(self) -> T:
        return cast(T, self._x.shape[0])

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[1])

    @property
    def sample_count(self) -> N:
        return cast(N, self._x.shape[2])

    @property
    def x_array(self) -> Float[JaxArray, "T K N"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "T K N"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "T K N"]:
        return self._heading

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_o, K, N]]:
        return np.stack([self._x, self._y, self._heading], axis=1)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstacleStates[T: int, K: int](
    ObstacleStates[T, K, JaxSampledObstacleStates[T, K, D[1]]]
):
    _x: Float[JaxArray, "T K"]
    _y: Float[JaxArray, "T K"]
    _heading: Float[JaxArray, "T K"]
    _covariance: ObstacleCovarianceArray[T, K] | None = None

    @staticmethod
    def empty[T_: int, K_: int = D[0]](
        *, horizon: T_, obstacle_count: K_ = 0
    ) -> "JaxObstacleStates[T_, K_]":
        """Creates obstacle states for zero obstacles over the given time horizon."""
        empty = jnp.full((horizon, obstacle_count), fill_value=jnp.nan)

        return JaxObstacleStates.create(
            x=empty,
            y=empty,
            heading=empty,
            horizon=horizon,
            obstacle_count=obstacle_count,
        )

    @staticmethod
    def sampled[N: int](  # type: ignore
        *,
        x: Float[JaxArray, "T K N"],
        y: Float[JaxArray, "T K N"],
        heading: Float[JaxArray, "T K N"],
        sample_count: N | None = None,
    ) -> JaxSampledObstacleStates[T, K, N]:
        return JaxSampledObstacleStates.create(
            x=x, y=y, heading=heading, sample_count=sample_count
        )

    @staticmethod
    def wrap[T_: int, K_: int](
        array: Float[JaxArray, f"T {D_O} K"],
        *,
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacleStates[T_, K_]":
        return JaxObstacleStates.create(
            x=array[:, 0, :], y=array[:, 1, :], heading=array[:, 2, :]
        )

    @staticmethod
    def create[T_: int, K_: int](
        *,
        x: Float[JaxArray, "T K"],
        y: Float[JaxArray, "T K"],
        heading: Float[JaxArray, "T K"],
        covariance: ObstacleCovarianceArray[T_, K_] | None = None,
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacleStates[T_, K_]":
        return JaxObstacleStates(_x=x, _y=y, _heading=heading, _covariance=covariance)

    @staticmethod
    def of_states[T_: int, K_: int](
        obstacle_states: Sequence["JaxObstacleStatesForTimeStep[K_]"],
        *,
        horizon: T_ | None = None,
    ) -> "JaxObstacleStates[T_, K_]":
        assert len(obstacle_states) > 0, "Obstacle states sequence must not be empty."
        assert horizon is None or len(obstacle_states) == horizon, (
            f"Expected horizon {horizon}, but got {len(obstacle_states)} obstacle states."
        )

        x = jnp.stack([states.x_array for states in obstacle_states], axis=0)
        y = jnp.stack([states.y_array for states in obstacle_states], axis=0)
        heading = jnp.stack(
            [states.heading_array for states in obstacle_states], axis=0
        )

        return JaxObstacleStates.create(x=x, y=y, heading=heading, horizon=horizon)

    @staticmethod
    def for_time_step[K_: int](
        *,
        x: Array[Dims[K_]] | Float[JaxArray, "K"],
        y: Array[Dims[K_]] | Float[JaxArray, "K"],
        heading: Array[Dims[K_]] | Float[JaxArray, "K"],
        obstacle_count: K_ | None = None,
        device: Device = "cpu",
    ) -> "JaxObstacleStatesForTimeStep[K_]":
        """Creates obstacle states for a single time step.

        Note:
            Since the common case is to further process this data on the CPU first,
            the default device is set to "cpu".
        """
        return JaxObstacleStatesForTimeStep.create(
            x=place(x, device=device),
            y=place(y, device=device),
            heading=place(heading, device=device),
            obstacle_count=obstacle_count,
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=1)

    def x(self) -> Array[Dims[T, K]]:
        return np.asarray(self._x)

    def y(self) -> Array[Dims[T, K]]:
        return np.asarray(self._y)

    def heading(self) -> Array[Dims[T, K]]:
        return np.asarray(self._heading)

    def covariance(self) -> Array[Dims[T, D_o, D_o, K]] | None:
        return np.asarray(self._covariance) if self._covariance is not None else None

    def positions(self) -> "JaxObstacle2dPositions[T, K]":
        return JaxObstacle2dPositions.create(x=self._x, y=self._y)

    def single(self) -> JaxSampledObstacleStates[T, K, D[1]]:
        return JaxSampledObstacleStates.create(
            x=self._x[..., jnp.newaxis],
            y=self._y[..., jnp.newaxis],
            heading=self._heading[..., jnp.newaxis],
        )

    def last(self) -> "JaxObstacleStatesForTimeStep[K]":
        return self.at(time_step=self.horizon - 1)

    def at(self, time_step: int) -> "JaxObstacleStatesForTimeStep[K]":
        return JaxObstacleStatesForTimeStep.create(
            x=self._x[time_step],
            y=self._y[time_step],
            heading=self._heading[time_step],
        )

    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        return jnp.stack([self._x, self._y, self._heading], axis=1)

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
    def covariance_array(self) -> ObstacleCovarianceArray[T, K] | None:
        return self._covariance

    @property
    def horizon(self) -> T:
        return cast(T, self._x.shape[0])

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[1])


@dataclass(kw_only=True, frozen=True)
class JaxObstacle2dPositions[T: int, K: int]:
    _x: Float[JaxArray, "T K"]
    _y: Float[JaxArray, "T K"]

    @staticmethod
    def create[T_: int, K_: int](
        *,
        x: Float[JaxArray, "T K"],
        y: Float[JaxArray, "T K"],
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacle2dPositions[T_, K_]":
        return JaxObstacle2dPositions(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[2], K]]:
        return self._numpy_array

    @property
    def horizon(self) -> T:
        return cast(T, self._x.shape[0])

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[1])

    @property
    def array(self) -> Float[JaxArray, "T 2 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, "T 2 K"]:
        return jnp.stack([self._x, self._y], axis=1)

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D[2], K]]:
        return np.stack([np.asarray(self._x), np.asarray(self._y)], axis=1)


@dataclass(kw_only=True, frozen=True)
class JaxObstacle2dPositionsForTimeStep[K: int]:
    _x: Float[JaxArray, "K"]
    _y: Float[JaxArray, "K"]

    @staticmethod
    def create[K_: int](
        *,
        x: Float[JaxArray, "K"],
        y: Float[JaxArray, "K"],
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacle2dPositionsForTimeStep[K_]":
        return JaxObstacle2dPositionsForTimeStep(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D[2], K]]:
        return self._numpy_array

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[0])

    @property
    def array(self) -> Float[JaxArray, "2 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, "2 K"]:
        return jnp.stack([self._x, self._y], axis=0)

    @cached_property
    def _numpy_array(self) -> Array[Dims[D[2], K]]:
        return np.stack([np.asarray(self._x), np.asarray(self._y)], axis=0)


@dataclass(kw_only=True, frozen=True)
class JaxObstacleStatesForTimeStep[K: int]:
    _x: Float[JaxArray, "K"]
    _y: Float[JaxArray, "K"]
    _heading: Float[JaxArray, "K"]

    @staticmethod
    def create[K_: int](
        *,
        x: Float[JaxArray, "K"],
        y: Float[JaxArray, "K"],
        heading: Float[JaxArray, "K"],
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacleStatesForTimeStep[K_]":
        return JaxObstacleStatesForTimeStep(_x=x, _y=y, _heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=0)

    def numpy(self) -> NumPyObstacleStatesForTimeStep[K]:
        return NumPyObstacleStatesForTimeStep.create(
            x=np.asarray(self._x),
            y=np.asarray(self._y),
            heading=np.asarray(self._heading),
        )

    def x(self) -> Array[Dims[K]]:
        return np.asarray(self._x)

    def y(self) -> Array[Dims[K]]:
        return np.asarray(self._y)

    def heading(self) -> Array[Dims[K]]:
        return np.asarray(self._heading)

    def positions(self) -> JaxObstacle2dPositionsForTimeStep[K]:
        return JaxObstacle2dPositionsForTimeStep.create(x=self._x, y=self._y)

    def replicate[T: int](self, *, horizon: T) -> JaxObstacleStates[T, K]:
        return JaxObstacleStates.create(
            x=jnp.tile(self._x[jnp.newaxis, :], (horizon, 1)),
            y=jnp.tile(self._y[jnp.newaxis, :], (horizon, 1)),
            heading=jnp.tile(self._heading[jnp.newaxis, :], (horizon, 1)),
        )

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[0])

    @property
    def x_array(self) -> Float[JaxArray, "K"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "K"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "K"]:
        return self._heading

    @property
    def array(self) -> Float[JaxArray, f"{D_O} K"]:
        return jnp.stack([self._x, self._y, self._heading], axis=0)
