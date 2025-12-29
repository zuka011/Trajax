from typing import Sequence, cast, Self, Any
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    jaxtyped,
    DataType,
    D_o,
    D_O,
    SampledObstacleStates,
    ObstacleStates,
    JaxObstacleStateProvider,
    ObstacleMotionPredictor,
)

from numtypes import Array, Dims, D
from jaxtyping import Array as JaxArray, Float, Scalar

import numpy as np
import jax
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

    @property
    def x_array(self) -> Float[JaxArray, "K"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "K"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "K"]:
        return self._heading


@dataclass(kw_only=True, frozen=True)
class JaxObstacleStatesRunningHistory[K: int]:
    history: list[JaxObstacleStatesForTimeStep[K]]

    @staticmethod
    def single[K_: int](
        step: JaxObstacleStatesForTimeStep[K_],
    ) -> "JaxObstacleStatesRunningHistory[K_]":
        return JaxObstacleStatesRunningHistory(history=[step])

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[int, D_o, K]]:
        return self._numpy_array

    def last(self) -> JaxObstacleStatesForTimeStep[K]:
        return self.history[-1]

    def append(self, step: JaxObstacleStatesForTimeStep[K]) -> Self:
        return self.__class__(history=self.history + [step])

    def x(self) -> Array[Dims[int, K]]:
        return self._x

    def y(self) -> Array[Dims[int, K]]:
        return self._y

    def heading(self) -> Array[Dims[int, K]]:
        return self._heading

    @property
    def horizon(self) -> int:
        return len(self.history)

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        if self.horizon == 0:
            return cast(K, 0)

        return cast(K, self.history[0]._x.shape[0])

    @cached_property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        return jnp.stack([self.x_array, self.y_array, self.heading_array], axis=1)

    @cached_property
    def x_array(self) -> Float[JaxArray, "T K"]:
        return jnp.stack([step._x for step in self.history], axis=0)

    @cached_property
    def y_array(self) -> Float[JaxArray, "T K"]:
        return jnp.stack([step._y for step in self.history], axis=0)

    @cached_property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        return jnp.stack([step._heading for step in self.history], axis=0)

    @cached_property
    def _numpy_array(self) -> Array[Dims[int, D_o, K]]:
        return np.stack([self._x, self._y, self._heading], axis=1)

    @cached_property
    def _x(self) -> Array[Dims[int, K]]:
        return np.stack([step._x for step in self.history], axis=0)

    @cached_property
    def _y(self) -> Array[Dims[int, K]]:
        return np.stack([step._y for step in self.history], axis=0)

    @cached_property
    def _heading(self) -> Array[Dims[int, K]]:
        return np.stack([step._heading for step in self.history], axis=0)


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

    def with_time_step(self, time_step: float) -> Self:
        return self

    def with_predictor(self, predictor: Any) -> Self:
        return self

    def __call__(self) -> JaxObstacleStates[T, K]:
        return self.states

    def step(self) -> None:
        pass


@dataclass(kw_only=True)
class JaxDynamicObstacleStateProvider[T: int, K: int](
    JaxObstacleStateProvider[JaxObstacleStates[T, K]]
):
    type MotionPredictor[T_: int, K_: int] = ObstacleMotionPredictor[
        JaxObstacleStatesRunningHistory[K_], JaxObstacleStates[T_, K_]
    ]

    history: JaxObstacleStatesRunningHistory[K]
    velocities: Float[JaxArray, "K 2"]

    horizon: T
    time_step: Scalar | None = None
    predictor: MotionPredictor[T, K] | None = None

    @staticmethod
    def create[T_: int, K_: int](
        *,
        positions: Float[JaxArray, "K 2"],
        velocities: Float[JaxArray, "K 2"],
        horizon: T_,
        obstacle_count: K_ | None = None,
    ) -> "JaxDynamicObstacleStateProvider[T_, K_]":
        headings = headings_from(velocities)

        return JaxDynamicObstacleStateProvider(
            history=JaxObstacleStatesRunningHistory.single(
                JaxObstacleStatesForTimeStep.create(
                    x=positions[:, 0], y=positions[:, 1], heading=headings
                )
            ),
            velocities=velocities,
            horizon=horizon,
        )

    def with_time_step(self, time_step: float) -> Self:
        return self.__class__(
            history=self.history,
            velocities=self.velocities,
            predictor=self.predictor,
            horizon=self.horizon,
            time_step=jnp.asarray(time_step),
        )

    def with_predictor(self, predictor: MotionPredictor) -> Self:
        return self.__class__(
            history=self.history,
            velocities=self.velocities,
            predictor=predictor,
            horizon=self.horizon,
            time_step=self.time_step,
        )

    def __call__(self) -> JaxObstacleStates[T, K]:
        assert self.predictor is not None, (
            "Motion predictor must be set to provide obstacle states."
        )

        return self.predictor.predict(history=self.history)

    def step(self) -> None:
        assert self.time_step is not None, (
            "Time step must be set to advance obstacle states."
        )

        last = self.history.last()
        x, y = step_obstacles(
            x=last.x_array,
            y=last.y_array,
            velocities=self.velocities,
            time_step=self.time_step,
        )
        self.history = self.history.append(
            JaxObstacleStatesForTimeStep.create(x=x, y=y, heading=last.heading_array)
        )


@jax.jit
@jaxtyped
def step_obstacles(
    *,
    x: Float[JaxArray, "K"],
    y: Float[JaxArray, "K"],
    velocities: Float[JaxArray, "K 2"],
    time_step: Scalar,
) -> tuple[Float[JaxArray, "K"], Float[JaxArray, "K"]]:
    new_x = x + velocities[:, 0] * time_step
    new_y = y + velocities[:, 1] * time_step
    return new_x, new_y


@jax.jit
@jaxtyped
def headings_from(velocities: Float[JaxArray, "K 2"]) -> Float[JaxArray, "K"]:
    speed = jnp.linalg.norm(velocities, axis=1)
    return jnp.where(speed > 1e-6, jnp.arctan2(velocities[:, 1], velocities[:, 0]), 0.0)
