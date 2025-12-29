from typing import Sequence, cast
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    DataType,
    D_o,
    D_O,
    SampledObstacleStates,
    ObstacleStates,
    JaxObstacleStateProvider,
)

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
class JaxObstacleStates[T: int, K: int](
    ObstacleStates[T, K, JaxSampledObstacleStates[T, K, D[1]]]
):
    _x: Float[JaxArray, "T K"]
    _y: Float[JaxArray, "T K"]
    _heading: Float[JaxArray, "T K"]
    _covariance: ObstacleCovarianceArray[T, K] | None = None

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
        obstacle_states: Sequence["JaxObstacleStates[int, K_]"],
        *,
        horizon: T_ | None = None,
    ) -> "JaxObstacleStates[T_, K_]":
        assert horizon is None or len(obstacle_states) == horizon

        x = jnp.stack([states.x_array[0] for states in obstacle_states], axis=0)
        y = jnp.stack([states.y_array[0] for states in obstacle_states], axis=0)
        heading = jnp.stack(
            [states.heading_array[0] for states in obstacle_states], axis=0
        )

        return JaxObstacleStates.create(x=x, y=y, heading=heading, horizon=horizon)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=-1)

    def x(self) -> Array[Dims[T, K]]:
        return np.asarray(self._x)

    def y(self) -> Array[Dims[T, K]]:
        return np.asarray(self._y)

    def heading(self) -> Array[Dims[T, K]]:
        return np.asarray(self._heading)

    def covariance(self) -> Array[Dims[T, D_o, D_o, K]] | None:
        return np.asarray(self._covariance) if self._covariance is not None else None

    def single(self) -> JaxSampledObstacleStates[T, K, D[1]]:
        return JaxSampledObstacleStates.create(
            x=self._x[..., jnp.newaxis],
            y=self._y[..., jnp.newaxis],
            heading=self._heading[..., jnp.newaxis],
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


@dataclass(frozen=True)
class JaxStaticObstacleStateProvider[T: int, K: int](
    JaxObstacleStateProvider[JaxObstacleStates[T, K]]
):
    states: JaxObstacleStates[T, K]

    @staticmethod
    def empty[T_: int](*, horizon: T_) -> "JaxStaticObstacleStateProvider[T_, D[0]]":
        positions = jnp.empty((0, 2))

        return JaxStaticObstacleStateProvider.create(
            positions=positions, horizon=horizon
        )

    @staticmethod
    def create[T_: int, K_: int](
        *,
        positions: Float[JaxArray, "K 2"],
        headings: Float[JaxArray, "K"] | None = None,
        horizon: T_,
        obstacle_count: K_ | None = None,
    ) -> "JaxStaticObstacleStateProvider[T_, K_]":
        K = obstacle_count if obstacle_count is not None else positions.shape[0]
        x = jnp.tile(positions[:, 0], (horizon, 1))
        y = jnp.tile(positions[:, 1], (horizon, 1))

        if headings is not None:
            heading = jnp.tile(headings, (horizon, 1))
        else:
            heading = jnp.zeros((horizon, K))

        assert x.shape == y.shape == heading.shape == (horizon, K), (
            f"Expected shapes {(horizon, K)}, but got x with shape {x.shape}, y with shape {y.shape}, heading with shape {heading.shape}."
        )

        return JaxStaticObstacleStateProvider(
            JaxObstacleStates.create(x=x, y=y, heading=heading)
        )

    def __call__(self) -> JaxObstacleStates[T, K]:
        return self.states
