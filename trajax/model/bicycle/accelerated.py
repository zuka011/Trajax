from typing import cast, Any
from dataclasses import dataclass

from trajax.type import jaxtyped
from trajax.model.bicycle.common import D_X, D_x, D_U, D_u, KinematicBicycleModel

from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims

import jax
import jax.numpy as jnp
import numpy as np


type StateArray = Float[JaxArray, f"{D_X}"]
type ControlInputArray = Float[JaxArray, f"{D_U}"]
type ControlInputSequenceArray = Float[JaxArray, f"T {D_U}"]

type StateBatchArray = Float[JaxArray, f"T {D_X} M"]
type ControlInputBatchArray = Float[JaxArray, f"T {D_U} M"]

type StatesAtTimeStep = Float[JaxArray, f"{D_X} M"]
type ControlInputsAtTimeStep = Float[JaxArray, f"{D_U} M"]


@jaxtyped
@dataclass(frozen=True)
class State:
    state: StateArray

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[D_x]]:
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


@dataclass(kw_only=True, frozen=True)
class StateSequence[T: int]:
    batch: "StateBatch[T, Any]"
    rollout: int

    def step(self, index: int) -> State:
        return State(self.batch.states[index, :, self.rollout])


@jaxtyped
@dataclass(frozen=True)
class StateBatch[T: int, M: int]:
    states: StateBatchArray

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, D_x, M]]:
        return np.asarray(self.states, dtype=dtype)

    def orientations(self) -> Array[Dims[T, M]]:
        return np.asarray(self.states[:, 2, :])

    def velocities(self) -> Array[Dims[T, M]]:
        return np.asarray(self.states[:, 3, :])

    def rollout(self, index: int) -> StateSequence[T]:
        return StateSequence(batch=self, rollout=index)

    @property
    def positions(self) -> "Positions[T, M]":
        return Positions(state=self)


@jaxtyped
@dataclass(frozen=True)
class Positions[T: int, M: int]:
    state: StateBatch

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, D_u, M]]:
        return np.asarray(self.state.states[:, :2, :], dtype=dtype)

    def x(self) -> Array[Dims[T, M]]:
        return np.asarray(self.state.states[:, 0, :])

    def y(self) -> Array[Dims[T, M]]:
        return np.asarray(self.state.states[:, 1, :])


@jaxtyped
@dataclass(frozen=True)
class ControlInputSequence[T: int]:
    inputs: ControlInputSequenceArray

    @staticmethod
    def zeroes[T_: int](horizon: T_) -> "ControlInputSequence[T_]":
        return ControlInputSequence(jnp.zeros((horizon, D_U)))

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, D_u]]:
        return np.asarray(self.inputs, dtype=dtype)

    @property
    def horizon(self) -> T:
        return cast(T, self.inputs.shape[0])

    @property
    def dimension(self) -> D_u:
        return cast(D_u, self.inputs.shape[1])


@jaxtyped
@dataclass(frozen=True)
class ControlInputBatch[T: int, M: int]:
    array: ControlInputBatchArray

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, D_u, M]]:
        return np.asarray(self.array, dtype=dtype)

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])


@dataclass(kw_only=True, frozen=True)
class JaxBicycleModel(
    KinematicBicycleModel[
        State, State, StateBatch, ControlInputSequence, ControlInputBatch
    ]
):
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
        self, inputs: ControlInputBatch[T, M], initial_state: State
    ) -> StateBatch[T, M]:
        rollout_count = inputs.rollout_count

        initial = jnp.stack(
            [
                jnp.full(rollout_count, initial_state.x),
                jnp.full(rollout_count, initial_state.y),
                jnp.full(rollout_count, initial_state.theta),
                jnp.full(rollout_count, initial_state.v),
            ]
        )

        return StateBatch(
            simulate(
                inputs.array,
                initial,
                time_step_size=self.time_step_size,
                wheelbase=self.wheelbase,
                speed_limits=self.speed_limits,
                steering_limits=self.steering_limits,
                acceleration_limits=self.acceleration_limits,
            )
        )

    async def step[T: int](self, input: ControlInputSequence[T], state: State) -> State:
        return State(
            step(
                state.state.reshape(-1, 1),
                input.inputs[0].reshape(-1, 1),
                time_step_size=self.time_step_size,
                wheelbase=self.wheelbase,
                speed_limits=self.speed_limits,
                steering_limits=self.steering_limits,
                acceleration_limits=self.acceleration_limits,
            )[:, 0]
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
def simulate(
    inputs: ControlInputBatchArray,
    initial: StatesAtTimeStep,
    *,
    time_step_size: Scalar,
    wheelbase: Scalar,
    speed_limits: tuple[Scalar, Scalar],
    steering_limits: tuple[Scalar, Scalar],
    acceleration_limits: tuple[Scalar, Scalar],
) -> StateBatchArray:
    @jax.jit
    @jaxtyped
    def do_step(
        state: StatesAtTimeStep, control: ControlInputsAtTimeStep
    ) -> tuple[StatesAtTimeStep, StatesAtTimeStep]:
        new_state = step(
            state,
            control,
            time_step_size=time_step_size,
            wheelbase=wheelbase,
            speed_limits=speed_limits,
            steering_limits=steering_limits,
            acceleration_limits=acceleration_limits,
        )
        return new_state, new_state

    _, states = jax.lax.scan(do_step, initial, inputs)
    return states


@jax.jit
@jaxtyped
def step(
    state: StatesAtTimeStep,
    control: ControlInputsAtTimeStep,
    *,
    time_step_size: Scalar,
    wheelbase: Scalar,
    speed_limits: tuple[Scalar, Scalar],
    steering_limits: tuple[Scalar, Scalar],
    acceleration_limits: tuple[Scalar, Scalar],
) -> StatesAtTimeStep:
    x, y, theta, v = state[0], state[1], state[2], state[3]
    a, delta = control[0], control[1]
    acceleration = jnp.clip(a, *acceleration_limits)
    steering = jnp.clip(delta, *steering_limits)

    new_x = x + v * jnp.cos(theta) * time_step_size
    new_y = y + v * jnp.sin(theta) * time_step_size
    new_theta = theta + v * jnp.tan(steering) / wheelbase * time_step_size
    new_v = jnp.clip(v + acceleration * time_step_size, *speed_limits)

    return jnp.stack([new_x, new_y, new_theta, new_v])
