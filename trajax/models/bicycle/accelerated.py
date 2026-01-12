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
    JaxBicycleObstacleStatesHistory,
    BicycleState,
    BicycleStateSequence,
    BicycleStateBatch,
    BicycleControlInputSequence,
    BicycleControlInputBatch,
    BicyclePositions,
    BicycleD_x,
    BICYCLE_D_X,
    BicycleD_u,
    BICYCLE_D_U,
    DynamicalModel,
    ObstacleModel,
    EstimatedObstacleStates,
)

from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims, D

import jax
import jax.numpy as jnp
import numpy as np

NO_LIMITS: Final = (jnp.asarray(-jnp.inf), jnp.asarray(jnp.inf))


type StateArray = Float[JaxArray, f"{BICYCLE_D_X}"]
type ControlInputSequenceArray[T: int] = Float[JaxArray, f"T {BICYCLE_D_U}"]
type StateBatchArray[T: int, M: int] = Float[JaxArray, f"T {BICYCLE_D_X} M"]
type ControlInputBatchArray[T: int, M: int] = Float[JaxArray, f"T {BICYCLE_D_U} M"]

type StatesAtTimeStep[M: int] = Float[JaxArray, f"{BICYCLE_D_X} M"]
type ControlInputsAtTimeStep[M: int] = Float[JaxArray, f"{BICYCLE_D_U} M"]


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleState(BicycleState, JaxState[BicycleD_x]):
    _array: StateArray

    @staticmethod
    def create(
        *,
        x: float | Scalar,
        y: float | Scalar,
        heading: float | Scalar,
        speed: float | Scalar,
    ) -> "JaxBicycleState":
        """Creates a JAX bicycle state from individual state components."""
        return JaxBicycleState(jnp.array([x, y, heading, speed]))

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[BicycleD_x]]:
        return np.asarray(self.array)

    @property
    def dimension(self) -> BicycleD_x:
        return cast(BicycleD_x, self.array.shape[0])

    @property
    def x(self) -> float:
        """Returns the x position of the bicycle."""
        return float(self.array[0])

    @property
    def y(self) -> float:
        """Returns the y position of the bicycle."""
        return float(self.array[1])

    @property
    def heading(self) -> float:
        """Returns the heading (orientation) of the bicycle."""
        return float(self.array[2])

    @property
    def speed(self) -> float:
        """Returns the speed (velocity magnitude) of the bicycle."""
        return float(self.array[3])

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

    @property
    def speed_scalar(self) -> Scalar:
        return self.array[3]


@dataclass(kw_only=True, frozen=True)
class JaxBicycleStateSequence[T: int, M: int = Any](
    BicycleStateSequence, JaxStateSequence[T, BicycleD_x]
):
    batch: "JaxBicycleStateBatch[T, M]"
    rollout: int

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[JaxBicycleState], *, horizon: T_ | None = None
    ) -> "JaxBicycleStateSequence[T_, D[1]]":
        """Creates a JAX bicycle state sequence from a sequence of bicycle states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        array = jnp.stack([state.array for state in states], axis=0)[:, :, jnp.newaxis]

        assert array.shape == (horizon, BICYCLE_D_X, 1), (
            f"Array shape {array.shape} does not match expected shape {(horizon, BICYCLE_D_X, 1)}."
        )

        return JaxBicycleStateSequence(
            batch=JaxBicycleStateBatch.wrap(array), rollout=0
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_x]]:
        return np.asarray(self.array)

    def step(self, index: int) -> JaxBicycleState:
        return JaxBicycleState(self.array[index])

    def batched(self) -> "JaxBicycleStateBatch[T, D[1]]":
        return JaxBicycleStateBatch.wrap(self.array[..., jnp.newaxis])

    def x(self) -> Array[Dims[T]]:
        return np.asarray(self.array[:, 0])

    def y(self) -> Array[Dims[T]]:
        return np.asarray(self.array[:, 1])

    def heading(self) -> Array[Dims[T]]:
        return np.asarray(self.array[:, 2])

    def speed(self) -> Array[Dims[T]]:
        return np.asarray(self.array[:, 3])

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> BicycleD_x:
        return cast(BicycleD_x, self.array.shape[1])

    @property
    def array(self) -> Float[JaxArray, f"T {BICYCLE_D_X}"]:
        return self.batch.array[:, :, self.rollout]


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleStateBatch[T: int, M: int](
    BicycleStateBatch[T, M], JaxStateBatch[T, BicycleD_x, M]
):
    _array: StateBatchArray[T, M]

    @staticmethod
    def wrap[T_: int, M_: int](
        array: StateBatchArray[T_, M_],
    ) -> "JaxBicycleStateBatch[T_, M_]":
        """Creates a JAX bicycle state batch from the given array."""
        return JaxBicycleStateBatch(array)

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[JaxBicycleState], *, horizon: T_ | None = None
    ) -> "JaxBicycleStateBatch[T_, int]":
        """Creates a bicycle state batch from a sequence of bicycle states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        array = jnp.stack([state.array for state in states], axis=0)[:, :, jnp.newaxis]

        assert array.shape == (expected := (horizon, BICYCLE_D_X, 1)), (
            f"Array shape {array.shape} does not match expected shape {expected}."
        )

        return JaxBicycleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_x, M]]:
        return np.asarray(self.array)

    def heading(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array[:, 2, :])

    def speed(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array[:, 3, :])

    def rollout(self, index: int) -> JaxBicycleStateSequence[T, M]:
        return JaxBicycleStateSequence(batch=self, rollout=index)

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> BicycleD_x:
        return cast(BicycleD_x, self.array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])

    @property
    def positions(self) -> "JaxBicyclePositions[T, M]":
        return JaxBicyclePositions(self)

    @property
    def array(self) -> StateBatchArray[T, M]:
        return self._array

    @property
    def heading_array(self) -> Float[JaxArray, "T M"]:
        return self.array[:, 2, :]


@jaxtyped
@dataclass(frozen=True)
class JaxBicyclePositions[T: int, M: int](BicyclePositions[T, M]):
    batch: JaxBicycleStateBatch[T, M]

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
class JaxBicycleControlInputSequence[T: int](
    BicycleControlInputSequence[T], JaxControlInputSequence[T, BicycleD_u]
):
    _array: ControlInputSequenceArray[T]

    @staticmethod
    def zeroes[T_: int](horizon: T_) -> "JaxBicycleControlInputSequence[T_]":
        return JaxBicycleControlInputSequence(jnp.zeros((horizon, BICYCLE_D_U)))

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_u]]:
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
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> BicycleD_u:
        return cast(BicycleD_u, self.array.shape[1])

    @property
    def array(self) -> ControlInputSequenceArray[T]:
        return self._array

    @property
    def accelerations_array(self) -> Float[JaxArray, "T"]:
        return self.array[:, 0]

    @property
    def steering_angles_array(self) -> Float[JaxArray, "T"]:
        return self.array[:, 1]


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleControlInputBatch[T: int, M: int](
    BicycleControlInputBatch[T, M], JaxControlInputBatch[T, BicycleD_u, M]
):
    _array: ControlInputBatchArray[T, M]

    @staticmethod
    def zero[T_: int, M_: int](
        *, horizon: T_, rollout_count: M_ = 1
    ) -> "JaxBicycleControlInputBatch[T_, M_]":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        array = jnp.zeros((horizon, BICYCLE_D_U, rollout_count))

        return JaxBicycleControlInputBatch(array)

    @staticmethod
    def create[T_: int = int, M_: int = int](
        *,
        array: ControlInputBatchArray[T_, M_],
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
    def of[T_: int = int](
        sequence: JaxBicycleControlInputSequence[T_],
    ) -> "JaxBicycleControlInputBatch[T_, D[1]]":
        """Creates a bicycle control input batch from a single control input sequence."""
        array = sequence.array[..., jnp.newaxis]

        assert array.shape == (expected := (sequence.horizon, BICYCLE_D_U, 1)), (
            f"Array shape {array.shape} does not match expected shape {expected}."
        )

        return JaxBicycleControlInputBatch(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_u, M]]:
        return np.asarray(self.array)

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> BicycleD_u:
        return cast(BicycleD_u, self.array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])

    @property
    def array(self) -> ControlInputBatchArray[T, M]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleObstacleStates[K: int]:
    array: Float[JaxArray, f"{BICYCLE_D_X} K"]

    @staticmethod
    def create[K_: int](
        *,
        x: Float[JaxArray, "K"],
        y: Float[JaxArray, "K"],
        heading: Float[JaxArray, "K"],
        speed: Float[JaxArray, "K"],
        count: K_ | None = None,
    ) -> "JaxBicycleObstacleStates[K_]":
        """Creates a JAX bicycle obstacle states from individual state components."""
        array = jnp.stack([x, y, heading, speed], axis=0)
        return JaxBicycleObstacleStates(array)


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleObstacleStateSequences[T: int, K: int]:
    array: Float[JaxArray, f"T {BICYCLE_D_X} K"]

    @staticmethod
    def create(
        *,
        x: Float[JaxArray, "T K"],
        y: Float[JaxArray, "T K"],
        heading: Float[JaxArray, "T K"],
        speed: Float[JaxArray, "T K"],
    ) -> "JaxBicycleObstacleStateSequences[int, int]":
        """Creates a JAX bicycle obstacle state sequences from individual state components."""
        array = jnp.stack([x, y, heading, speed], axis=1)
        return JaxBicycleObstacleStateSequences(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_x, K]]:
        return np.asarray(self.array)

    def x(self) -> Array[Dims[T, K]]:
        return np.asarray(self.array[:, 0, :])

    def y(self) -> Array[Dims[T, K]]:
        return np.asarray(self.array[:, 1, :])

    def heading(self) -> Array[Dims[T, K]]:
        return np.asarray(self.array[:, 2, :])

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> BicycleD_x:
        return cast(BicycleD_x, self.array.shape[1])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[2])

    @property
    def x_array(self) -> Float[JaxArray, "T K"]:
        return self.array[:, 0, :]

    @property
    def y_array(self) -> Float[JaxArray, "T K"]:
        return self.array[:, 1, :]

    @property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        return self.array[:, 2, :]


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleObstacleVelocities[K: int]:
    steering_angles: Float[JaxArray, "K"]

    @property
    def count(self) -> K:
        return cast(K, self.steering_angles.shape[0])


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleObstacleControlInputSequences[T: int, K: int]:
    array: Float[JaxArray, f"T {BICYCLE_D_U} K"]

    @staticmethod
    def create(
        *,
        accelerations: Float[JaxArray, "T K"],
        steering_angles: Float[JaxArray, "T K"],
    ) -> "JaxBicycleObstacleControlInputSequences[int, int]":
        """Creates a JAX bicycle obstacle control input sequences from individual input components."""
        array = jnp.stack([accelerations, steering_angles], axis=1)
        return JaxBicycleObstacleControlInputSequences(array)


@dataclass(kw_only=True, frozen=True)
class JaxBicycleModel(
    DynamicalModel[
        JaxBicycleState,
        JaxBicycleStateSequence,
        JaxBicycleStateBatch,
        JaxBicycleControlInputSequence,
        JaxBicycleControlInputBatch,
    ],
):
    _time_step_size: float
    time_step_size_scalar: Scalar
    wheelbase: Scalar
    speed_limits: tuple[Scalar, Scalar]
    steering_limits: tuple[Scalar, Scalar]
    acceleration_limits: tuple[Scalar, Scalar]

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

        return JaxBicycleModel(
            _time_step_size=time_step_size,
            time_step_size_scalar=jnp.asarray(time_step_size),
            wheelbase=jnp.asarray(wheelbase),
            speed_limits=wrap(speed_limits) if speed_limits is not None else NO_LIMITS,
            steering_limits=wrap(steering_limits)
            if steering_limits is not None
            else NO_LIMITS,
            acceleration_limits=wrap(acceleration_limits)
            if acceleration_limits is not None
            else NO_LIMITS,
        )

    def simulate[T: int, M: int](
        self, inputs: JaxBicycleControlInputBatch[T, M], initial_state: JaxBicycleState
    ) -> JaxBicycleStateBatch[T, M]:
        rollout_count = inputs.rollout_count

        initial = jnp.stack(
            [
                jnp.full(rollout_count, initial_state.x_scalar),
                jnp.full(rollout_count, initial_state.y_scalar),
                jnp.full(rollout_count, initial_state.heading_scalar),
                jnp.full(rollout_count, initial_state.speed_scalar),
            ]
        )

        return JaxBicycleStateBatch(
            simulate(
                inputs.array,
                initial,
                time_step_size=self.time_step_size_scalar,
                wheelbase=self.wheelbase,
                speed_limits=self.speed_limits,
                steering_limits=self.steering_limits,
                acceleration_limits=self.acceleration_limits,
            )
        )

    def step[T: int](
        self, inputs: JaxBicycleControlInputSequence[T], state: JaxBicycleState
    ) -> JaxBicycleState:
        return JaxBicycleState(
            step(
                state.array.reshape(-1, 1),
                inputs.array[0].reshape(-1, 1),
                time_step_size=self.time_step_size_scalar,
                wheelbase=self.wheelbase,
                speed_limits=self.speed_limits,
                steering_limits=self.steering_limits,
                acceleration_limits=self.acceleration_limits,
            )[:, 0]
        )

    def forward[T: int](
        self, inputs: JaxBicycleControlInputSequence[T], state: JaxBicycleState
    ) -> JaxBicycleStateSequence[T]:
        return self.simulate(JaxBicycleControlInputBatch.of(inputs), state).rollout(0)

    @property
    def time_step_size(self) -> float:
        return self._time_step_size


@dataclass(kw_only=True, frozen=True)
class JaxBicycleObstacleModel(
    ObstacleModel[
        JaxBicycleObstacleStatesHistory,
        JaxBicycleObstacleStates,
        JaxBicycleObstacleVelocities,
        JaxBicycleObstacleControlInputSequences,
        JaxBicycleObstacleStateSequences,
    ]
):
    time_step_size: Scalar
    wheelbase: Scalar

    @staticmethod
    def create(
        *, time_step_size: float, wheelbase: float = 1.0
    ) -> "JaxBicycleObstacleModel":
        """Creates a JAX bicycle obstacle model."""
        return JaxBicycleObstacleModel(
            time_step_size=jnp.asarray(time_step_size), wheelbase=jnp.asarray(wheelbase)
        )

    def estimate_state_from[K: int](
        self, history: JaxBicycleObstacleStatesHistory[int, K]
    ) -> EstimatedObstacleStates[
        JaxBicycleObstacleStates[K], JaxBicycleObstacleVelocities[K]
    ]:
        assert history.horizon > 0, "History must have at least one time step."

        speeds, steering_angles = estimate_velocities(
            x_history=history.x_array,
            y_history=history.y_array,
            heading_history=history.heading_array,
            time_step_size=self.time_step_size,
            wheelbase=self.wheelbase,
        )

        return EstimatedObstacleStates(
            states=JaxBicycleObstacleStates.create(
                x=history.x_array[-1],
                y=history.y_array[-1],
                heading=history.heading_array[-1],
                speed=speeds,
            ),
            velocities=JaxBicycleObstacleVelocities(steering_angles=steering_angles),
        )

    def input_to_maintain[K: int](
        self,
        velocities: JaxBicycleObstacleVelocities[K],
        *,
        states: JaxBicycleObstacleStates[K],
        horizon: int,
    ) -> JaxBicycleObstacleControlInputSequences[int, K]:
        return JaxBicycleObstacleControlInputSequences.create(  # type: ignore[return-value]
            accelerations=jnp.zeros((horizon, velocities.count)),
            steering_angles=jnp.tile(
                velocities.steering_angles[jnp.newaxis, :], (horizon, 1)
            ),
        )

    def forward[T: int, K: int](
        self,
        *,
        current: JaxBicycleObstacleStates[K],
        inputs: JaxBicycleObstacleControlInputSequences[T, K],
    ) -> JaxBicycleObstacleStateSequences[T, K]:
        return JaxBicycleObstacleStateSequences(
            simulate(
                inputs.array,
                current.array,
                time_step_size=self.time_step_size,
                wheelbase=self.wheelbase,
                speed_limits=NO_LIMITS,
                steering_limits=NO_LIMITS,
                acceleration_limits=NO_LIMITS,
            )
        )


def wrap(limits: tuple[float, float]) -> tuple[Scalar, Scalar]:
    return (jnp.asarray(limits[0]), jnp.asarray(limits[1]))


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


@jax.jit
@jaxtyped
def estimate_velocities(
    *,
    x_history: Float[JaxArray, "T K"],
    y_history: Float[JaxArray, "T K"],
    heading_history: Float[JaxArray, "T K"],
    time_step_size: Scalar,
    wheelbase: Scalar,
) -> tuple[Float[JaxArray, "K"], Float[JaxArray, "K"]]:
    horizon = x_history.shape[0]
    count = x_history.shape[1]
    return jax.lax.cond(
        horizon > 1,
        lambda: _estimate_velocities_with_two_points(
            x_history=x_history,
            y_history=y_history,
            heading_history=heading_history,
            time_step_size=time_step_size,
            wheelbase=wheelbase,
        ),
        lambda: (jnp.zeros(count), jnp.zeros(count)),
    )


@jax.jit
@jaxtyped
def _estimate_velocities_with_two_points(
    *,
    x_history: Float[JaxArray, "T K"],
    y_history: Float[JaxArray, "T K"],
    heading_history: Float[JaxArray, "T K"],
    time_step_size: Scalar,
    wheelbase: Scalar,
) -> tuple[Float[JaxArray, "K"], Float[JaxArray, "K"]]:
    delta_x = x_history[-1] - x_history[-2]
    delta_y = y_history[-1] - y_history[-2]

    speeds = jnp.sqrt(delta_x**2 + delta_y**2) / time_step_size

    heading_velocity = (heading_history[-1] - heading_history[-2]) / time_step_size

    steering_angles = jnp.where(
        speeds > 1e-6,
        jnp.arctan(heading_velocity * wheelbase / speeds),
        jnp.zeros_like(speeds),
    )

    return speeds, steering_angles
