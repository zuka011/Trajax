from typing import cast
from dataclasses import dataclass

from trajax.type import jaxtyped
from trajax.bicycle.common import D_X, D_x, D_U, D_u

from jaxtyping import Array, Float, Scalar
from numtypes import Array as NumPyArray, Dims

import jax
import jax.numpy as jnp
import numpy as np


type StateArray = Float[Array, f"{D_X}"]
type ControlInputArray = Float[Array, f"{D_U}"]
type ControlInputSequenceArray = Float[Array, f"T {D_U}"]

type StateBatchArray = Float[Array, f"T {D_X} M"]
type ControlInputBatchArray = Float[Array, f"T {D_U} M"]


@jaxtyped
@dataclass(frozen=True)
class JaxState:
    state: StateArray

    def __array__(self, dtype: np.dtype | None = None) -> NumPyArray[Dims[D_x]]:
        return np.asarray(self.state, dtype=dtype)

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def theta(self) -> float:
        return float(self.state[2])

    @property
    def v(self) -> float:
        return float(self.state[3])


@jaxtyped
@dataclass(frozen=True)
class JaxStateBatch[T: int, M: int]:
    states: StateBatchArray

    def __array__(self, dtype: np.dtype | None = None) -> NumPyArray[Dims[T, D_x, M]]:
        return np.asarray(self.states, dtype=dtype)

    def orientations(self) -> NumPyArray[Dims[T, M]]:
        return np.asarray(self.states[:, 2, :])

    def velocities(self) -> NumPyArray[Dims[T, M]]:
        return np.asarray(self.states[:, 3, :])

    @property
    def positions(self) -> "JaxPositions[T, M]":
        return JaxPositions(state=self)


@jaxtyped
@dataclass(frozen=True)
class JaxPositions[T: int, M: int]:
    state: JaxStateBatch

    def __array__(self, dtype: np.dtype | None = None) -> NumPyArray[Dims[T, D_u, M]]:
        return np.asarray(self.state.states[:, :2, :], dtype=dtype)

    def x(self) -> NumPyArray[Dims[T, M]]:
        return np.asarray(self.state.states[:, 0, :])

    def y(self) -> NumPyArray[Dims[T, M]]:
        return np.asarray(self.state.states[:, 1, :])


@jaxtyped
@dataclass(frozen=True)
class JaxControlInputSequence[T: int]:
    inputs: ControlInputSequenceArray

    def __array__(self, dtype: np.dtype | None = None) -> NumPyArray[Dims[T, D_u]]:
        return np.asarray(self.inputs, dtype=dtype)


@jaxtyped
@dataclass(frozen=True)
class JaxControlInputBatch[T: int, M: int]:
    inputs: ControlInputBatchArray

    def __array__(self, dtype: np.dtype | None = None) -> NumPyArray[Dims[T, D_u, M]]:
        return np.asarray(self.inputs, dtype=dtype)

    @property
    def rollout_count(self) -> M:
        return cast(M, self.inputs.shape[2])

    @property
    def horizon(self) -> T:
        return cast(T, self.inputs.shape[0])


@dataclass(kw_only=True, frozen=True)
class JaxBicycleModel:
    time_step_size: float
    wheelbase: float
    speed_limits: tuple[float, float]
    steering_limits: tuple[float, float]
    acceleration_limits: tuple[float, float]

    @staticmethod
    def create(
        *,
        time_step_size: float,
        wheelbase: float = 1.0,
        speed_limits: tuple[float, float] | None = None,
        steering_limits: tuple[float, float] | None = None,
        acceleration_limits: tuple[float, float] | None = None,
    ) -> "JaxBicycleModel":
        """Creates a kinematic bicycle model that uses JAX for computations."""
        no_limits = (float("-inf"), float("inf"))

        return JaxBicycleModel(
            time_step_size=time_step_size,
            wheelbase=wheelbase,
            speed_limits=speed_limits if speed_limits is not None else no_limits,
            steering_limits=steering_limits
            if steering_limits is not None
            else no_limits,
            acceleration_limits=acceleration_limits
            if acceleration_limits is not None
            else no_limits,
        )

    async def simulate[T: int, M: int](
        self,
        inputs: JaxControlInputBatch[T, M],
        initial_state: JaxState,
    ) -> JaxStateBatch[T, M]:
        rollout_count = inputs.rollout_count

        initial = jnp.stack(
            [
                jnp.full(rollout_count, initial_state.x),
                jnp.full(rollout_count, initial_state.y),
                jnp.full(rollout_count, initial_state.theta),
                jnp.full(rollout_count, initial_state.v),
            ]
        )

        return JaxStateBatch(
            simulate_jit(
                inputs.inputs,
                initial,
                time_step_size=self.time_step_size,
                wheelbase=self.wheelbase,
                speed_limits=self.speed_limits,
                steering_limits=self.steering_limits,
                acceleration_limits=self.acceleration_limits,
            )
        )

    async def step[T: int](
        self, input: JaxControlInputSequence[T], state: JaxState
    ) -> JaxState:
        raise NotImplementedError(
            "Single step simulation is not implemented for JaxBicycleModel."
        )

    @property
    def min_speed(self) -> float:
        return self.speed_limits[0]

    @property
    def max_speed(self) -> float:
        return self.speed_limits[1]

    @property
    def min_steering(self) -> float:
        return self.steering_limits[0]

    @property
    def max_steering(self) -> float:
        return self.steering_limits[1]

    @property
    def min_acceleration(self) -> float:
        return self.acceleration_limits[0]

    @property
    def max_acceleration(self) -> float:
        return self.acceleration_limits[1]


@jax.jit
@jaxtyped
def simulate_jit(
    inputs: ControlInputBatchArray,
    initial: Float[Array, "4 M"],
    *,
    time_step_size: Scalar,
    wheelbase: Scalar,
    speed_limits: tuple[Scalar, Scalar],
    steering_limits: tuple[Scalar, Scalar],
    acceleration_limits: tuple[Scalar, Scalar],
) -> StateBatchArray:
    @jax.jit
    @jaxtyped
    def step(
        state: Float[Array, "4 M"], control: Float[Array, "2 M"]
    ) -> tuple[Float[Array, "4 M"], Float[Array, "4 M"]]:
        x, y, theta, v = state[0], state[1], state[2], state[3]
        acceleration = jnp.clip(control[0], *acceleration_limits)
        steering = jnp.clip(control[1], *steering_limits)

        new_x = x + v * jnp.cos(theta) * time_step_size
        new_y = y + v * jnp.sin(theta) * time_step_size
        new_theta = theta + v * jnp.tan(steering) / wheelbase * time_step_size
        new_v = jnp.clip(v + acceleration * time_step_size, *speed_limits)

        new_state = jnp.stack([new_x, new_y, new_theta, new_v])
        return new_state, new_state

    _, states = jax.lax.scan(step, initial, inputs)
    return states
