from typing import Self, overload, cast, Sequence, Final, Any
from dataclasses import dataclass

from trajax.types import (
    DataType,
    NumPyState,
    NumPyStateSequence,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyBicycleObstacleStatesHistory,
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

from numtypes import Array, Dims, D, shape_of, array

import numpy as np

NO_LIMITS: Final = (float("-inf"), float("inf"))

type StateArray = Array[Dims[BicycleD_x]]
type ControlInputSequenceArray[T: int] = Array[Dims[T, BicycleD_u]]
type StateBatchArray[T: int, M: int] = Array[Dims[T, BicycleD_x, M]]
type ControlInputBatchArray[T: int, M: int] = Array[Dims[T, BicycleD_u, M]]

type StatesAtTimeStep[M: int] = Array[Dims[BicycleD_x, M]]
type ControlInputsAtTimeStep[M: int] = Array[Dims[BicycleD_u, M]]


@dataclass(frozen=True)
class NumPyBicycleState(BicycleState, NumPyState[BicycleD_x]):
    _array: StateArray

    @staticmethod
    def create(
        *, x: float, y: float, heading: float, speed: float
    ) -> "NumPyBicycleState":
        """Creates a NumPy bicycle state from individual state components."""
        return NumPyBicycleState(array([x, y, heading, speed], shape=(BICYCLE_D_X,)))

    def __array__(self, dtype: DataType | None = None) -> StateArray:
        return self.array

    @property
    def dimension(self) -> BicycleD_x:
        return self.array.shape[0]

    @property
    def x(self) -> float:
        """Returns the x position of the bicycle."""
        return self.array[0]

    @property
    def y(self) -> float:
        """Returns the y position of the bicycle."""
        return self.array[1]

    @property
    def heading(self) -> float:
        """Returns the heading (orientation) of the bicycle."""
        return self.array[2]

    @property
    def speed(self) -> float:
        """Returns the speed (velocity magnitude) of the bicycle."""
        return self.array[3]

    @property
    def array(self) -> StateArray:
        return self._array


@dataclass(kw_only=True, frozen=True)
class NumPyBicycleStateSequence[T: int, M: int = Any](
    BicycleStateSequence, NumPyStateSequence[T, BicycleD_x]
):
    batch: "NumPyBicycleStateBatch[T, M]"
    rollout: int

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[NumPyBicycleState], *, horizon: T_ | None = None
    ) -> "NumPyBicycleStateSequence[T_, D[1]]":
        """Creates a NumPy bicycle state sequence from a sequence of bicycle states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        assert shape_of(array, matches=(horizon, BICYCLE_D_X, 1))

        return NumPyBicycleStateSequence(
            batch=NumPyBicycleStateBatch.wrap(array), rollout=0
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_x]]:
        return self.array

    def step(self, index: int) -> NumPyBicycleState:
        return NumPyBicycleState(self.array[index])

    def batched(self) -> "NumPyBicycleStateBatch[T, D[1]]":
        return NumPyBicycleStateBatch.wrap(self.array[..., np.newaxis])

    def x(self) -> Array[Dims[T]]:
        return self.array[:, 0]

    def y(self) -> Array[Dims[T]]:
        return self.array[:, 1]

    def heading(self) -> Array[Dims[T]]:
        return self.array[:, 2]

    def speed(self) -> Array[Dims[T]]:
        return self.array[:, 3]

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_x:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[T, BicycleD_x]]:
        return self.batch.array[:, :, self.rollout]


@dataclass(frozen=True)
class NumPyBicycleStateBatch[T: int, M: int](
    BicycleStateBatch[T, M], NumPyStateBatch[T, BicycleD_x, M]
):
    _array: StateBatchArray[T, M]

    @staticmethod
    def wrap[T_: int, M_: int](
        array: StateBatchArray[T_, M_],
    ) -> "NumPyBicycleStateBatch[T_, M_]":
        """Creates a NumPy bicycle state batch from the given array."""
        return NumPyBicycleStateBatch(array)

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[NumPyBicycleState], *, horizon: T_ | None = None
    ) -> "NumPyBicycleStateBatch[int, D[1]]":
        """Creates a NumPy bicycle state batch from a sequence of bicycle states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        assert shape_of(array, matches=(horizon, BICYCLE_D_X, 1))

        return NumPyBicycleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> StateBatchArray[T, M]:
        return self.array

    def heading(self) -> Array[Dims[T, M]]:
        return self.array[:, 2, :]

    def speed(self) -> Array[Dims[T, M]]:
        return self.array[:, 3, :]

    def rollout(self, index: int) -> NumPyBicycleStateSequence[T, M]:
        return NumPyBicycleStateSequence(batch=self, rollout=index)

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_x:
        return self.array.shape[1]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]

    @property
    def positions(self) -> "NumPyBicyclePositions[T, M]":
        return NumPyBicyclePositions(batch=self)

    @property
    def array(self) -> StateBatchArray[T, M]:
        return self._array


@dataclass(frozen=True)
class NumPyBicyclePositions[T: int, M: int](BicyclePositions[T, M]):
    batch: NumPyBicycleStateBatch[T, M]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[2], M]]:
        return self.batch.array[:, :2, :]

    def x(self) -> Array[Dims[T, M]]:
        return self.batch.array[:, 0, :]

    def y(self) -> Array[Dims[T, M]]:
        return self.batch.array[:, 1, :]


@dataclass(frozen=True)
class NumPyBicycleControlInputSequence[T: int](
    BicycleControlInputSequence[T], NumPyControlInputSequence[T, BicycleD_u]
):
    _array: ControlInputSequenceArray[T]

    @staticmethod
    def zeroes[T_: int](horizon: T_) -> "NumPyBicycleControlInputSequence[T_]":
        """Creates a zeroed control input sequence for the given horizon."""
        array = np.zeros((horizon, BICYCLE_D_U))

        assert shape_of(array, matches=(horizon, BICYCLE_D_U))

        return NumPyBicycleControlInputSequence(array)

    def __array__(self, dtype: DataType | None = None) -> ControlInputSequenceArray[T]:
        return self.array

    @overload
    def similar(self, *, array: Array[Dims[T, BicycleD_u]]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Array[Dims[L, BicycleD_u]], length: L
    ) -> "NumPyBicycleControlInputSequence[L]": ...

    def similar[L: int](
        self, *, array: Array[Dims[L, BicycleD_u]], length: L | None = None
    ) -> "Self | NumPyBicycleControlInputSequence[L]":
        # NOTE: "Wrong" cast to silence the type checker.
        effective_length = cast(T, length if length is not None else array.shape[0])

        assert shape_of(
            array, matches=(effective_length, self.dimension), name="similar array"
        )

        return self.__class__(array)

    def accelerations(self) -> Array[Dims[T]]:
        return self.array[:, 0]

    def steering_angles(self) -> Array[Dims[T]]:
        return self.array[:, 1]

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_u:
        return self.array.shape[1]

    @property
    def array(self) -> ControlInputSequenceArray[T]:
        return self._array


@dataclass(frozen=True)
class NumPyBicycleControlInputBatch[T: int, M: int](
    BicycleControlInputBatch[T, M], NumPyControlInputBatch[T, BicycleD_u, M]
):
    _array: ControlInputBatchArray[T, M]

    @staticmethod
    def zero[T_: int, M_: int](
        *, horizon: T_, rollout_count: M_ = 1
    ) -> "NumPyBicycleControlInputBatch[T_, M_]":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        array = np.zeros((horizon, BICYCLE_D_U, rollout_count))

        assert shape_of(array, matches=(horizon, BICYCLE_D_U, rollout_count))

        return NumPyBicycleControlInputBatch(array)

    @staticmethod
    def create[T_: int, M_: int](
        *, array: Array[Dims[T_, BicycleD_u, M_]]
    ) -> "NumPyBicycleControlInputBatch[T_, M_]":
        """Creates a NumPy bicycle control input batch from the given array."""

        return NumPyBicycleControlInputBatch(array)

    @staticmethod
    def of[T_: int](
        sequence: NumPyBicycleControlInputSequence[T_],
    ) -> "NumPyBicycleControlInputBatch[T_, D[1]]":
        """Creates a NumPy bicycle control input batch from a single control input sequence."""
        array = sequence.array[..., np.newaxis]

        assert shape_of(array, matches=(sequence.horizon, BICYCLE_D_U, 1))

        return NumPyBicycleControlInputBatch(array)

    def __array__(self, dtype: DataType | None = None) -> ControlInputBatchArray[T, M]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_u:
        return self.array.shape[1]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]

    @property
    def array(self) -> ControlInputBatchArray[T, M]:
        return self._array


@dataclass(frozen=True)
class NumPyBicycleObstacleStates[K: int]:
    array: Array[Dims[BicycleD_x, K]]

    @staticmethod
    def create(
        *,
        x: Array[Dims[K]],
        y: Array[Dims[K]],
        heading: Array[Dims[K]],
        speed: Array[Dims[K]],
    ) -> "NumPyBicycleObstacleStates[K]":
        """Creates a NumPy bicycle obstacle states from individual state components."""
        array = np.stack([x, y, heading, speed], axis=0)

        assert shape_of(array, matches=(BICYCLE_D_X, x.shape[0]))

        return NumPyBicycleObstacleStates(array)


@dataclass(frozen=True)
class NumPyBicycleObstacleStateSequences[T: int, K: int]:
    array: Array[Dims[T, BicycleD_x, K]]

    @staticmethod
    def create(
        *,
        x: Array[Dims[T, K]],
        y: Array[Dims[T, K]],
        heading: Array[Dims[T, K]],
        speed: Array[Dims[T, K]],
    ) -> "NumPyBicycleObstacleStateSequences[T, K]":
        """Creates a NumPy bicycle obstacle state sequences from individual state components."""
        T, K = x.shape
        array = np.stack([x, y, heading, speed], axis=1)

        assert shape_of(array, matches=(T, BICYCLE_D_X, K))

        return NumPyBicycleObstacleStateSequences(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_x, K]]:
        return self.array

    def x(self) -> Array[Dims[T, K]]:
        return self.array[:, 0, :]

    def y(self) -> Array[Dims[T, K]]:
        return self.array[:, 1, :]

    def heading(self) -> Array[Dims[T, K]]:
        return self.array[:, 2, :]

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_x:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]


@dataclass(frozen=True)
class NumPyBicycleObstacleVelocities[K: int]:
    steering_angles: Array[Dims[K]]

    @property
    def count(self) -> K:
        return self.steering_angles.shape[0]


@dataclass(frozen=True)
class NumPyBicycleObstacleControlInputSequences[T: int, K: int]:
    array: Array[Dims[T, BicycleD_u, K]]

    @staticmethod
    def create(
        *,
        accelerations: Array[Dims[T, K]],
        steering_angles: Array[Dims[T, K]],
    ) -> "NumPyBicycleObstacleControlInputSequences[T, K]":
        """Creates a NumPy bicycle obstacle control input sequences from individual input components."""
        T, K = accelerations.shape
        array = np.stack([accelerations, steering_angles], axis=1)

        assert shape_of(array, matches=(T, BICYCLE_D_U, K))

        return NumPyBicycleObstacleControlInputSequences(array)


@dataclass(kw_only=True, frozen=True)
class NumPyBicycleModel(
    DynamicalModel[
        NumPyBicycleState,
        NumPyBicycleStateSequence,
        NumPyBicycleStateBatch,
        NumPyBicycleControlInputSequence,
        NumPyBicycleControlInputBatch,
    ],
):
    _time_step_size: float
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
    ) -> "NumPyBicycleModel":
        """Creates a kinematic bicycle model that uses NumPy for computations."""

        return NumPyBicycleModel(
            _time_step_size=time_step_size,
            wheelbase=wheelbase,
            speed_limits=speed_limits if speed_limits is not None else NO_LIMITS,
            steering_limits=steering_limits
            if steering_limits is not None
            else NO_LIMITS,
            acceleration_limits=acceleration_limits
            if acceleration_limits is not None
            else NO_LIMITS,
        )

    def simulate[T: int, M: int](
        self,
        inputs: NumPyBicycleControlInputBatch[T, M],
        initial_state: NumPyBicycleState,
    ) -> NumPyBicycleStateBatch[T, M]:
        rollout_count = inputs.rollout_count

        initial = np.stack(
            [
                np.full(rollout_count, initial_state.x),
                np.full(rollout_count, initial_state.y),
                np.full(rollout_count, initial_state.heading),
                np.full(rollout_count, initial_state.speed),
            ]
        )

        return NumPyBicycleStateBatch(
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
        self, inputs: NumPyBicycleControlInputSequence[T], state: NumPyBicycleState
    ) -> NumPyBicycleState:
        state_as_rollouts = state.array.reshape(-1, 1)
        first_input = inputs.array[0].reshape(-1, 1)

        assert shape_of(
            state_as_rollouts, matches=(BICYCLE_D_X, 1), name="state reshaped for step"
        )
        assert shape_of(
            first_input,
            matches=(BICYCLE_D_U, 1),
            name="first control input reshaped for step",
        )

        return NumPyBicycleState(
            step(
                state_as_rollouts,
                first_input,
                time_step_size=self.time_step_size,
                wheelbase=self.wheelbase,
                speed_limits=self.speed_limits,
                steering_limits=self.steering_limits,
                acceleration_limits=self.acceleration_limits,
            )[:, 0]
        )

    def forward[T: int](
        self, inputs: NumPyBicycleControlInputSequence[T], state: NumPyBicycleState
    ) -> NumPyBicycleStateSequence[T]:
        return self.simulate(NumPyBicycleControlInputBatch.of(inputs), state).rollout(0)

    @property
    def time_step_size(self) -> float:
        return self._time_step_size


@dataclass(kw_only=True, frozen=True)
class NumPyBicycleObstacleModel(
    ObstacleModel[
        NumPyBicycleObstacleStatesHistory,
        NumPyBicycleObstacleStates,
        NumPyBicycleObstacleVelocities,
        NumPyBicycleObstacleControlInputSequences,
        NumPyBicycleObstacleStateSequences,
    ]
):
    time_step_size: float
    wheelbase: float

    @staticmethod
    def create(
        *, time_step_size: float, wheelbase: float = 1.0
    ) -> "NumPyBicycleObstacleModel":
        """Creates a NumPy bicycle obstacle model."""
        return NumPyBicycleObstacleModel(
            time_step_size=time_step_size, wheelbase=wheelbase
        )

    def estimate_state_from[K: int](
        self, history: NumPyBicycleObstacleStatesHistory[int, K]
    ) -> EstimatedObstacleStates[
        NumPyBicycleObstacleStates[K], NumPyBicycleObstacleVelocities[K]
    ]:
        assert history.horizon > 0, "History must have at least one time step."

        if history.horizon < 2:
            speeds = np.zeros((history.count))
            steering_angles = np.zeros((history.count))
        else:
            speeds = estimate_speeds_from(history, time_step_size=self.time_step_size)
            steering_angles = estimate_steering_angles_from(
                history,
                speeds=speeds,
                time_step_size=self.time_step_size,
                wheelbase=self.wheelbase,
            )

        assert shape_of(speeds, matches=(history.count,))
        assert shape_of(steering_angles, matches=(history.count,))

        return EstimatedObstacleStates(
            states=NumPyBicycleObstacleStates.create(
                x=history.x()[-1],
                y=history.y()[-1],
                heading=history.heading()[-1, :],
                speed=speeds,
            ),
            velocities=NumPyBicycleObstacleVelocities(steering_angles=steering_angles),
        )

    def input_to_maintain[K: int](
        self,
        velocities: NumPyBicycleObstacleVelocities[K],
        *,
        states: NumPyBicycleObstacleStates[K],
        horizon: int,
    ) -> NumPyBicycleObstacleControlInputSequences[int, K]:
        accelerations = np.zeros((horizon, velocities.count))

        assert shape_of(accelerations, matches=(horizon, velocities.count))

        return NumPyBicycleObstacleControlInputSequences.create(
            accelerations=accelerations,
            steering_angles=np.tile(
                velocities.steering_angles[np.newaxis, :], (horizon, 1)
            ),
        )

    def forward[T: int, K: int](
        self,
        *,
        current: NumPyBicycleObstacleStates[K],
        inputs: NumPyBicycleObstacleControlInputSequences[T, K],
    ) -> NumPyBicycleObstacleStateSequences[T, K]:
        result = simulate(
            inputs.array,
            current.array,
            time_step_size=self.time_step_size,
            wheelbase=self.wheelbase,
            speed_limits=(float("-inf"), float("inf")),
            steering_limits=(float("-inf"), float("inf")),
            acceleration_limits=(float("-inf"), float("inf")),
        )

        return NumPyBicycleObstacleStateSequences.create(
            x=result[:, 0, :],
            y=result[:, 1, :],
            heading=result[:, 2, :],
            speed=result[:, 3, :],
        )


def simulate[T: int, N: int](
    inputs: ControlInputBatchArray[T, N],
    initial: StatesAtTimeStep[N],
    *,
    time_step_size: float,
    wheelbase: float,
    speed_limits: tuple[float, float],
    steering_limits: tuple[float, float],
    acceleration_limits: tuple[float, float],
) -> StateBatchArray[T, N]:
    horizon = inputs.shape[0]
    rollout_count = inputs.shape[2]
    states = np.zeros((horizon, BICYCLE_D_X, rollout_count))
    current = initial

    for t in range(horizon):
        current = step(
            current,
            inputs[t],
            time_step_size=time_step_size,
            wheelbase=wheelbase,
            speed_limits=speed_limits,
            steering_limits=steering_limits,
            acceleration_limits=acceleration_limits,
        )
        states[t] = current

    assert shape_of(
        states, matches=(horizon, BICYCLE_D_X, rollout_count), name="simulated states"
    )

    return states


def step[M: int](
    state: StatesAtTimeStep[M],
    control: ControlInputsAtTimeStep[M],
    *,
    time_step_size: float,
    wheelbase: float,
    speed_limits: tuple[float, float],
    steering_limits: tuple[float, float],
    acceleration_limits: tuple[float, float],
) -> StatesAtTimeStep[M]:
    x, y, theta, v = state[0], state[1], state[2], state[3]
    a, delta = control[0], control[1]
    acceleration = np.clip(a, *acceleration_limits)
    steering = np.clip(delta, *steering_limits)

    new_x = x + v * np.cos(theta) * time_step_size
    new_y = y + v * np.sin(theta) * time_step_size
    new_theta = theta + v * np.tan(steering) / wheelbase * time_step_size
    new_v = np.clip(v + acceleration * time_step_size, *speed_limits)

    return np.stack([new_x, new_y, new_theta, new_v])


def estimate_speeds_from[K: int](
    history: NumPyBicycleObstacleStatesHistory[int, K], *, time_step_size: float
) -> Array[Dims[K]]:
    assert history.horizon >= 2, (
        "At least two history steps are required to estimate speed."
    )

    delta_x = history.x()[-1] - history.x()[-2]
    delta_y = history.y()[-1] - history.y()[-2]

    speeds = np.sqrt(delta_x**2 + delta_y**2) / time_step_size

    assert shape_of(speeds, matches=(history.count,), name="estimated speeds")

    return speeds


def estimate_steering_angles_from[K: int](
    history: NumPyBicycleObstacleStatesHistory[int, K],
    *,
    speeds: Array[Dims[K]],
    time_step_size: float,
    wheelbase: float,
) -> Array[Dims[K]]:
    assert history.horizon >= 2, (
        "At least two history steps are required to estimate heading."
    )

    heading_velocity = (
        history.heading()[-1, :] - history.heading()[-2, :]
    ) / time_step_size

    with np.errstate(invalid="ignore"):
        steering_angles = np.where(
            speeds > 1e-6, np.arctan(heading_velocity * wheelbase / speeds), 0.0
        )

    assert shape_of(
        steering_angles, matches=(history.count,), name="estimated steering angles"
    )

    return steering_angles
