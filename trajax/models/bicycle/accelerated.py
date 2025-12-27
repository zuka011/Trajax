from typing import cast, Any, overload, Self, Sequence
from dataclasses import dataclass

from trajax.type import jaxtyped
from trajax.mppi import (
    JaxState,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
)
from trajax.models.bicycle.common import (
    BICYCLE_D_X,
    BicycleD_x,
    BICYCLE_D_U,
    BicycleD_u,
    BicycleModel,
)

from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims

import jax
import jax.numpy as jnp
import numpy as np


type StateArray = Float[JaxArray, f"{BICYCLE_D_X}"]
type ControlInputArray = Float[JaxArray, f"{BICYCLE_D_U}"]
type ControlInputSequenceArray = Float[JaxArray, f"T {BICYCLE_D_U}"]

type StateBatchArray = Float[JaxArray, f"T {BICYCLE_D_X} M"]
type ControlInputBatchArray = Float[JaxArray, f"T {BICYCLE_D_U} M"]

type StatesAtTimeStep = Float[JaxArray, f"{BICYCLE_D_X} M"]
type ControlInputsAtTimeStep = Float[JaxArray, f"{BICYCLE_D_U} M"]


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleState(JaxState[BicycleD_x]):
    _array: StateArray

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[BicycleD_x]]:
        return np.asarray(self.array)

    @property
    def array(self) -> StateArray:
        return self._array

    @property
    def dimension(self) -> BicycleD_x:
        return cast(BicycleD_x, self.array.shape[0])

    @property
    def x(self) -> float:
        return float(self.array[0])

    @property
    def y(self) -> float:
        return float(self.array[1])

    @property
    def theta(self) -> float:
        return float(self.array[2])

    @property
    def v(self) -> float:
        return float(self.array[3])

    @property
    def x_scalar(self) -> Scalar:
        return self.array[0]

    @property
    def y_scalar(self) -> Scalar:
        return self.array[1]

    @property
    def theta_scalar(self) -> Scalar:
        return self.array[2]

    @property
    def v_scalar(self) -> Scalar:
        return self.array[3]


@dataclass(kw_only=True, frozen=True)
class StateSequence:
    batch: "JaxBicycleStateBatch[Any, Any]"
    rollout: int

    def step(self, index: int) -> JaxBicycleState:
        return JaxBicycleState(self.batch.array[index, :, self.rollout])


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleStateBatch[T: int, M: int](JaxStateBatch[T, BicycleD_x, M]):
    _array: StateBatchArray

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[JaxBicycleState], *, horizon: T_ | None = None
    ) -> "JaxBicycleStateBatch[T_, int]":
        """Creates a bicycle state batch from a sequence of bicycle states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))

        array = jnp.stack([state.array for state in states], axis=0)[:, :, jnp.newaxis]

        return JaxBicycleStateBatch(array)

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, BicycleD_x, M]]:
        return np.asarray(self.array)

    def orientations(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array[:, 2, :])

    def velocities(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array[:, 3, :])

    def rollout(self, index: int) -> StateSequence:
        return StateSequence(batch=self, rollout=index)

    @property
    def array(self) -> StateBatchArray:
        return self._array

    @property
    def positions(self) -> "JaxBicyclePositions[T, M]":
        return JaxBicyclePositions(array=self)

    @property
    def orientations_array(self) -> Float[JaxArray, "T M"]:
        return self.array[:, 2, :]

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> BicycleD_x:
        return cast(BicycleD_x, self.array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])


@jaxtyped
@dataclass(frozen=True)
class JaxBicyclePositions[T: int, M: int]:
    array: JaxBicycleStateBatch

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, BicycleD_u, M]]:
        return np.asarray(self.array.array[:, :2, :])

    def x(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array.array[:, 0, :])

    def y(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array.array[:, 1, :])

    @property
    def x_array(self) -> Float[JaxArray, "T M"]:
        return self.array.array[:, 0, :]

    @property
    def y_array(self) -> Float[JaxArray, "T M"]:
        return self.array.array[:, 1, :]


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleControlInputSequence[T: int](JaxControlInputSequence[T, BicycleD_u]):
    _array: ControlInputSequenceArray

    @staticmethod
    def zeroes[T_: int](horizon: T_) -> "JaxBicycleControlInputSequence[T_]":
        return JaxBicycleControlInputSequence(jnp.zeros((horizon, BICYCLE_D_U)))

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, BicycleD_u]]:
        return np.asarray(self.array)

    @overload
    def similar(self, *, array: Float[JaxArray, "T D_u"]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L
    ) -> "JaxBicycleControlInputSequence[L]": ...

    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L | None = None
    ) -> Self | "JaxBicycleControlInputSequence[L]":
        expected_length = length if length is not None else array.shape[0]

        assert array.shape == (expected := (expected_length, BICYCLE_D_U)), (
            f"Array shape {array.shape} does not match expected shape {expected}."
        )

        return self.__class__(array)

    @property
    def array(self) -> ControlInputSequenceArray:
        return self._array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> BicycleD_u:
        return cast(BicycleD_u, self.array.shape[1])


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleControlInputBatch[T: int, M: int](
    JaxControlInputBatch[T, BicycleD_u, M]
):
    _array: ControlInputBatchArray

    @staticmethod
    def create[T_: int = int, M_: int = int](
        *,
        array: ControlInputBatchArray,
        horizon: T_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "JaxBicycleControlInputBatch[T_, M_]":
        horizon = horizon if horizon is not None else cast(T_, array.shape[0])
        rollout_count = (
            rollout_count if rollout_count is not None else cast(M_, array.shape[2])
        )

        assert array.shape == (expected := (horizon, BICYCLE_D_U, rollout_count)), (
            f"Array shape {array.shape} does not match expected shape {expected}."
        )

        return JaxBicycleControlInputBatch(array)

    @staticmethod
    def zero[T_: int, M_: int](
        *, horizon: T_, rollout_count: M_ = 1
    ) -> "JaxBicycleControlInputBatch[T_, M_]":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        array = jnp.zeros((horizon, BICYCLE_D_U, rollout_count))

        return JaxBicycleControlInputBatch(array)

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, BicycleD_u, M]]:
        return np.asarray(self.array)

    @property
    def array(self) -> ControlInputBatchArray:
        return self._array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> BicycleD_u:
        return cast(BicycleD_u, self.array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])


@dataclass(kw_only=True, frozen=True)
class JaxBicycleModel(
    BicycleModel[
        JaxBicycleState,
        JaxBicycleStateBatch,
        JaxBicycleControlInputSequence,
        JaxBicycleControlInputBatch,
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

    def simulate[T: int, M: int](
        self, inputs: JaxBicycleControlInputBatch[T, M], initial_state: JaxBicycleState
    ) -> JaxBicycleStateBatch[T, M]:
        rollout_count = inputs.rollout_count

        initial = jnp.stack(
            [
                jnp.full(rollout_count, initial_state.x_scalar),
                jnp.full(rollout_count, initial_state.y_scalar),
                jnp.full(rollout_count, initial_state.theta_scalar),
                jnp.full(rollout_count, initial_state.v_scalar),
            ]
        )

        return JaxBicycleStateBatch(
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

    def step[T: int](
        self, input: JaxBicycleControlInputSequence[T], state: JaxBicycleState
    ) -> JaxBicycleState:
        return JaxBicycleState(
            step(
                state.array.reshape(-1, 1),
                input.array[0].reshape(-1, 1),
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
