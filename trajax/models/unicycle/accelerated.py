from typing import cast, overload, Self, Sequence, Final, Any
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    DataType,
    JaxState,
    JaxStateSequence,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
    UnicycleState,
    UnicycleStateSequence,
    UnicycleStateBatch,
    UnicycleControlInputSequence,
    UnicycleControlInputBatch,
    UnicyclePositions,
    UnicycleD_x,
    UNICYCLE_D_X,
    UnicycleD_u,
    UNICYCLE_D_U,
    DynamicalModel,
)

from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims, D

import jax
import jax.numpy as jnp
import numpy as np

NO_LIMITS: Final = (jnp.asarray(-jnp.inf), jnp.asarray(jnp.inf))


type StateArray = Float[JaxArray, f"{UNICYCLE_D_X}"]
type ControlInputSequenceArray[T: int] = Float[JaxArray, f"T {UNICYCLE_D_U}"]
type StateBatchArray[T: int, M: int] = Float[JaxArray, f"T {UNICYCLE_D_X} M"]
type ControlInputBatchArray[T: int, M: int] = Float[JaxArray, f"T {UNICYCLE_D_U} M"]

type StatesAtTimeStep[M: int] = Float[JaxArray, f"{UNICYCLE_D_X} M"]
type ControlInputsAtTimeStep[M: int] = Float[JaxArray, f"{UNICYCLE_D_U} M"]


@jaxtyped
@dataclass(frozen=True)
class JaxUnicycleState(UnicycleState, JaxState[UnicycleD_x]):
    _array: StateArray

    @staticmethod
    def create(
        *,
        x: float | Scalar,
        y: float | Scalar,
        heading: float | Scalar,
    ) -> "JaxUnicycleState":
        """Creates a JAX unicycle state from individual state components."""
        return JaxUnicycleState(jnp.array([x, y, heading]))

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[UnicycleD_x]]:
        return np.asarray(self.array)

    @property
    def dimension(self) -> UnicycleD_x:
        return cast(UnicycleD_x, self.array.shape[0])

    @property
    def x(self) -> float:
        """Returns the x position of the unicycle."""
        return float(self.array[0])

    @property
    def y(self) -> float:
        """Returns the y position of the unicycle."""
        return float(self.array[1])

    @property
    def heading(self) -> float:
        """Returns the heading (orientation) of the unicycle."""
        return float(self.array[2])

    @property
    def array(self) -> StateArray:
        return self._array

    @property
    def x_scalar(self) -> Scalar:
        return self.array[0]

    @property
    def y_scalar(self) -> Scalar:
        return self.array[1]

    @property
    def heading_scalar(self) -> Scalar:
        return self.array[2]


@dataclass(kw_only=True, frozen=True)
class JaxUnicycleStateSequence[T: int, M: int = Any](
    UnicycleStateSequence, JaxStateSequence[T, UnicycleD_x]
):
    batch: "JaxUnicycleStateBatch[T, M]"
    rollout: int

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[JaxUnicycleState], *, horizon: T_ | None = None
    ) -> "JaxUnicycleStateSequence[T_, D[1]]":
        """Creates a JAX unicycle state sequence from a sequence of unicycle states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        array = jnp.stack([state.array for state in states], axis=0)[:, :, jnp.newaxis]

        assert array.shape == (horizon, UNICYCLE_D_X, 1), (
            f"Array shape {array.shape} does not match expected shape {(horizon, UNICYCLE_D_X, 1)}."
        )

        return JaxUnicycleStateSequence(
            batch=JaxUnicycleStateBatch.wrap(array), rollout=0
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, UnicycleD_x]]:
        return np.asarray(self.array)

    def step(self, index: int) -> JaxUnicycleState:
        return JaxUnicycleState(self.array[index])

    def batched(self) -> "JaxUnicycleStateBatch[T, D[1]]":
        return JaxUnicycleStateBatch.wrap(self.array[..., jnp.newaxis])

    def x(self) -> Array[Dims[T]]:
        return np.asarray(self.array[:, 0])

    def y(self) -> Array[Dims[T]]:
        return np.asarray(self.array[:, 1])

    def heading(self) -> Array[Dims[T]]:
        return np.asarray(self.array[:, 2])

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> UnicycleD_x:
        return cast(UnicycleD_x, self.array.shape[1])

    @property
    def array(self) -> Float[JaxArray, f"T {UNICYCLE_D_X}"]:
        return self.batch.array[:, :, self.rollout]


@jaxtyped
@dataclass(frozen=True)
class JaxUnicycleStateBatch[T: int, M: int](
    UnicycleStateBatch[T, M], JaxStateBatch[T, UnicycleD_x, M]
):
    _array: StateBatchArray[T, M]

    @staticmethod
    def wrap[T_: int, M_: int](
        array: StateBatchArray[T_, M_],
    ) -> "JaxUnicycleStateBatch[T_, M_]":
        """Creates a JAX unicycle state batch from the given array."""
        return JaxUnicycleStateBatch(array)

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[JaxUnicycleState], *, horizon: T_ | None = None
    ) -> "JaxUnicycleStateBatch[T_, int]":
        """Creates a unicycle state batch from a sequence of unicycle states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        array = jnp.stack([state.array for state in states], axis=0)[:, :, jnp.newaxis]

        assert array.shape == (expected := (horizon, UNICYCLE_D_X, 1)), (
            f"Array shape {array.shape} does not match expected shape {expected}."
        )

        return JaxUnicycleStateBatch(array)

    def __array__(
        self, dtype: DataType | None = None
    ) -> Array[Dims[T, UnicycleD_x, M]]:
        return np.asarray(self.array)

    def heading(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array[:, 2, :])

    def rollout(self, index: int) -> JaxUnicycleStateSequence[T, M]:
        return JaxUnicycleStateSequence(batch=self, rollout=index)

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> UnicycleD_x:
        return cast(UnicycleD_x, self.array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])

    @property
    def positions(self) -> "JaxUnicyclePositions[T, M]":
        return JaxUnicyclePositions(self)

    @property
    def array(self) -> StateBatchArray[T, M]:
        return self._array

    @property
    def heading_array(self) -> Float[JaxArray, "T M"]:
        return self.array[:, 2, :]


@jaxtyped
@dataclass(frozen=True)
class JaxUnicyclePositions[T: int, M: int](UnicyclePositions[T, M]):
    batch: JaxUnicycleStateBatch[T, M]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[2], M]]:
        return np.asarray(self.batch.array[:, :2, :])

    def x(self) -> Array[Dims[T, M]]:
        return np.asarray(self.batch.array[:, 0, :])

    def y(self) -> Array[Dims[T, M]]:
        return np.asarray(self.batch.array[:, 1, :])

    @property
    def x_array(self) -> Float[JaxArray, "T M"]:
        return self.batch.array[:, 0, :]

    @property
    def y_array(self) -> Float[JaxArray, "T M"]:
        return self.batch.array[:, 1, :]


@jaxtyped
@dataclass(frozen=True)
class JaxUnicycleControlInputSequence[T: int](
    UnicycleControlInputSequence[T], JaxControlInputSequence[T, UnicycleD_u]
):
    _array: ControlInputSequenceArray[T]

    @staticmethod
    def zeroes[T_: int](horizon: T_) -> "JaxUnicycleControlInputSequence[T_]":
        return JaxUnicycleControlInputSequence(jnp.zeros((horizon, UNICYCLE_D_U)))

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, UnicycleD_u]]:
        return np.asarray(self.array)

    @overload
    def similar(self, *, array: Float[JaxArray, "T D_u"]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L
    ) -> "JaxUnicycleControlInputSequence[L]": ...

    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L | None = None
    ) -> Self | "JaxUnicycleControlInputSequence[L]":
        expected_length = length if length is not None else array.shape[0]

        assert array.shape == (expected := (expected_length, UNICYCLE_D_U)), (
            f"Array shape {array.shape} does not match expected shape {expected}."
        )

        return self.__class__(array)

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> UnicycleD_u:
        return cast(UnicycleD_u, self.array.shape[1])

    @property
    def array(self) -> ControlInputSequenceArray[T]:
        return self._array

    @property
    def linear_velocity_array(self) -> Float[JaxArray, "T"]:
        return self.array[:, 0]

    @property
    def angular_velocity_array(self) -> Float[JaxArray, "T"]:
        return self.array[:, 1]


@jaxtyped
@dataclass(frozen=True)
class JaxUnicycleControlInputBatch[T: int, M: int](
    UnicycleControlInputBatch[T, M], JaxControlInputBatch[T, UnicycleD_u, M]
):
    _array: ControlInputBatchArray[T, M]

    @staticmethod
    def zero[T_: int, M_: int](
        *, horizon: T_, rollout_count: M_ = 1
    ) -> "JaxUnicycleControlInputBatch[T_, M_]":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        array = jnp.zeros((horizon, UNICYCLE_D_U, rollout_count))

        return JaxUnicycleControlInputBatch(array)

    @staticmethod
    def create[T_: int = int, M_: int = int](
        *,
        array: ControlInputBatchArray[T_, M_],
        horizon: T_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "JaxUnicycleControlInputBatch[T_, M_]":
        horizon = horizon if horizon is not None else cast(T_, array.shape[0])
        rollout_count = (
            rollout_count if rollout_count is not None else cast(M_, array.shape[2])
        )

        assert array.shape == (expected := (horizon, UNICYCLE_D_U, rollout_count)), (
            f"Array shape {array.shape} does not match expected shape {expected}."
        )

        return JaxUnicycleControlInputBatch(array)

    @staticmethod
    def of[T_: int = int](
        sequence: JaxUnicycleControlInputSequence[T_],
    ) -> "JaxUnicycleControlInputBatch[T_, D[1]]":
        """Creates a unicycle control input batch from a single control input sequence."""
        array = sequence.array[..., jnp.newaxis]

        assert array.shape == (expected := (sequence.horizon, UNICYCLE_D_U, 1)), (
            f"Array shape {array.shape} does not match expected shape {expected}."
        )

        return JaxUnicycleControlInputBatch(array)

    def __array__(
        self, dtype: DataType | None = None
    ) -> Array[Dims[T, UnicycleD_u, M]]:
        return np.asarray(self.array)

    def linear_velocity(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array[:, 0, :])

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> UnicycleD_u:
        return cast(UnicycleD_u, self.array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])

    @property
    def array(self) -> ControlInputBatchArray[T, M]:
        return self._array


@dataclass(kw_only=True, frozen=True)
class JaxUnicycleModel(
    DynamicalModel[
        JaxUnicycleState,
        JaxUnicycleStateSequence,
        JaxUnicycleStateBatch,
        JaxUnicycleControlInputSequence,
        JaxUnicycleControlInputBatch,
    ],
):
    _time_step_size: float
    time_step_size_scalar: Scalar
    speed_limits: tuple[Scalar, Scalar]
    angular_velocity_limits: tuple[Scalar, Scalar]

    @staticmethod
    def create(
        *,
        time_step_size: float,
        speed_limits: tuple[float, float] | None = None,
        angular_velocity_limits: tuple[float, float] | None = None,
    ) -> "JaxUnicycleModel":
        """Creates a unicycle model that uses JAX for computations."""

        return JaxUnicycleModel(
            _time_step_size=time_step_size,
            time_step_size_scalar=jnp.asarray(time_step_size),
            speed_limits=wrap(speed_limits) if speed_limits is not None else NO_LIMITS,
            angular_velocity_limits=wrap(angular_velocity_limits)
            if angular_velocity_limits is not None
            else NO_LIMITS,
        )

    def simulate[T: int, M: int](
        self,
        inputs: JaxUnicycleControlInputBatch[T, M],
        initial_state: JaxUnicycleState,
    ) -> JaxUnicycleStateBatch[T, M]:
        rollout_count = inputs.rollout_count

        initial = jnp.stack(
            [
                jnp.full(rollout_count, initial_state.x_scalar),
                jnp.full(rollout_count, initial_state.y_scalar),
                jnp.full(rollout_count, initial_state.heading_scalar),
            ]
        )

        return JaxUnicycleStateBatch(
            simulate(
                inputs.array,
                initial,
                time_step_size=self.time_step_size_scalar,
                speed_limits=self.speed_limits,
                angular_velocity_limits=self.angular_velocity_limits,
            )
        )

    def step[T: int](
        self, inputs: JaxUnicycleControlInputSequence[T], state: JaxUnicycleState
    ) -> JaxUnicycleState:
        return JaxUnicycleState(
            step(
                state.array.reshape(-1, 1),
                inputs.array[0].reshape(-1, 1),
                time_step_size=self.time_step_size_scalar,
                speed_limits=self.speed_limits,
                angular_velocity_limits=self.angular_velocity_limits,
            )[:, 0]
        )

    def forward[T: int](
        self, inputs: JaxUnicycleControlInputSequence[T], state: JaxUnicycleState
    ) -> JaxUnicycleStateSequence[T]:
        return self.simulate(JaxUnicycleControlInputBatch.of(inputs), state).rollout(0)

    @property
    def time_step_size(self) -> float:
        return self._time_step_size


def wrap(limits: tuple[float, float]) -> tuple[Scalar, Scalar]:
    return (jnp.asarray(limits[0]), jnp.asarray(limits[1]))


@jax.jit
@jaxtyped
def simulate(
    inputs: ControlInputBatchArray,
    initial: StatesAtTimeStep,
    *,
    time_step_size: Scalar,
    speed_limits: tuple[Scalar, Scalar],
    angular_velocity_limits: tuple[Scalar, Scalar],
) -> StateBatchArray:
    def do_step(
        state: StatesAtTimeStep, control: ControlInputsAtTimeStep
    ) -> tuple[StatesAtTimeStep, StatesAtTimeStep]:
        new_state = step(
            state,
            control,
            time_step_size=time_step_size,
            speed_limits=speed_limits,
            angular_velocity_limits=angular_velocity_limits,
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
    speed_limits: tuple[Scalar, Scalar],
    angular_velocity_limits: tuple[Scalar, Scalar],
) -> StatesAtTimeStep:
    x, y, theta = state[0], state[1], state[2]
    v, omega = control[0], control[1]
    linear_velocity = jnp.clip(v, *speed_limits)
    angular_velocity = jnp.clip(omega, *angular_velocity_limits)

    new_x = x + linear_velocity * jnp.cos(theta) * time_step_size
    new_y = y + linear_velocity * jnp.sin(theta) * time_step_size
    new_theta = theta + angular_velocity * time_step_size

    return jnp.stack([new_x, new_y, new_theta])
