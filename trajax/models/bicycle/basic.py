from typing import Self, overload, cast, Sequence
from dataclasses import dataclass

from trajax.types import (
    DataType,
    NumPyState,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyBicycleObstacleStatesHistory,
    NumPyObstacleStates,
    BicycleState,
    BicycleStateSequence,
    BicycleStateBatch,
    BicycleControlInputSequence,
    BicycleControlInputBatch,
    BicyclePositions,
    BicycleD_x,
    BICYCLE_D_X,
    BicycleD_v,
    BICYCLE_D_V,
    BicycleD_u,
    BICYCLE_D_U,
    DynamicalModel,
    ObstacleModel,
    EstimatedObstacleStates,
)
from trajax.obstacles import NumPyObstaclePositionsAndHeadings

from numtypes import Array, Dims, D, shape_of, array

import numpy as np


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
    def theta(self) -> float:
        """Returns the orientation (heading) of the bicycle."""
        return self.array[2]

    @property
    def v(self) -> float:
        """Returns the speed (velocity magnitude) of the bicycle."""
        return self.array[3]

    @property
    def array(self) -> StateArray:
        return self._array


@dataclass(kw_only=True, frozen=True)
class NumPyBicycleStateSequence[T: int, M: int](BicycleStateSequence):
    batch: "NumPyBicycleStateBatch[T, M]"
    rollout: int

    def step(self, index: int) -> NumPyBicycleState:
        return NumPyBicycleState(self.batch.array[index, :, self.rollout])


@dataclass(frozen=True)
class NumPyBicycleStateBatch[T: int, M: int](
    BicycleStateBatch[T, M], NumPyStateBatch[T, BicycleD_x, M]
):
    _array: StateBatchArray[T, M]

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[NumPyBicycleState], *, horizon: T_ | None = None
    ) -> "NumPyBicycleStateBatch[int, D[1]]":
        """Creates a NumPy bicycle state batch from a sequence of bicycle states."""
        horizon = horizon if horizon is not None else cast(T_, len(states))

        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        assert shape_of(array, matches=(horizon, BICYCLE_D_X, 1))

        return NumPyBicycleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> StateBatchArray[T, M]:
        return self.array

    def orientations(self) -> Array[Dims[T, M]]:
        return self.array[:, 2, :]

    def velocities(self) -> Array[Dims[T, M]]:
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
    _array: Array[Dims[BicycleD_x, K]]

    @staticmethod
    def create(
        *,
        x: Array[Dims[K]],
        y: Array[Dims[K]],
        theta: Array[Dims[K]],
        v: Array[Dims[K]],
    ) -> "NumPyBicycleObstacleStates[K]":
        """Creates a NumPy bicycle obstacle states from individual state components."""
        array = np.stack([x, y, theta, v], axis=0)

        assert shape_of(array, matches=(BICYCLE_D_X, x.shape[0]))

        return NumPyBicycleObstacleStates(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[BicycleD_x, K]]:
        return self.array

    @property
    def dimension(self) -> BicycleD_x:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[BicycleD_x, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyBicycleObstacleVelocities[K: int]:
    _array: Array[Dims[BicycleD_v, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[BicycleD_v, K]]:
        return self.array

    @property
    def dimension(self) -> BicycleD_v:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[BicycleD_v, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyBicycleObstacleControlInputSequences[T: int, K: int]:
    _array: Array[Dims[T, BicycleD_u, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_u, K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_u:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, BicycleD_u, K]]:
        return self._array


@dataclass(kw_only=True, frozen=True)
class NumPyBicycleModel(
    DynamicalModel[
        NumPyBicycleState,
        NumPyBicycleStateBatch,
        NumPyBicycleControlInputSequence,
        NumPyBicycleControlInputBatch,
    ],
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
    ) -> "NumPyBicycleModel":
        """Creates a kinematic bicycle model that uses NumPy for computations."""
        no_limits = (float("-inf"), float("inf"))

        return NumPyBicycleModel(
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
        self,
        inputs: NumPyBicycleControlInputBatch[T, M],
        initial_state: NumPyBicycleState,
    ) -> NumPyBicycleStateBatch[T, M]:
        rollout_count = inputs.rollout_count

        initial = np.stack(
            [
                np.full(rollout_count, initial_state.x),
                np.full(rollout_count, initial_state.y),
                np.full(rollout_count, initial_state.theta),
                np.full(rollout_count, initial_state.v),
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
        self, input: NumPyBicycleControlInputSequence[T], state: NumPyBicycleState
    ) -> NumPyBicycleState:
        state_as_rollouts = state.array.reshape(-1, 1)
        first_input = input.array[0].reshape(-1, 1)

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


@dataclass(kw_only=True, frozen=True)
class NumPyBicycleObstacleModel(
    ObstacleModel[
        NumPyBicycleObstacleStatesHistory,
        NumPyBicycleObstacleStates,
        NumPyBicycleObstacleVelocities,
        NumPyBicycleObstacleControlInputSequences,
        NumPyObstacleStates,
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
        if history.horizon < 2:
            velocities = np.zeros((BICYCLE_D_V, history.count))
        else:
            velocities = np.stack(
                [
                    estimate_heading_from(history, time_step_size=self.time_step_size),
                    estimate_speed_from(history, time_step_size=self.time_step_size),
                ],
                axis=0,
            )

        assert shape_of(velocities, matches=(BICYCLE_D_V, history.count))

        return EstimatedObstacleStates(
            states=NumPyBicycleObstacleStates.create(
                x=history.x()[-1],
                y=history.y()[-1],
                theta=history.heading()[-1, :],
                v=velocities[1, :],
            ),
            velocities=NumPyBicycleObstacleVelocities(velocities),
        )

    def input_to_maintain[K: int](
        self,
        velocities: NumPyBicycleObstacleVelocities[K],
        *,
        states: NumPyBicycleObstacleStates[K],
        horizon: int,
    ) -> NumPyBicycleObstacleControlInputSequences[int, K]:
        inputs = np.zeros((horizon, BICYCLE_D_U, velocities.count))

        assert shape_of(inputs, matches=(horizon, BICYCLE_D_U, velocities.count))

        return NumPyBicycleObstacleControlInputSequences(inputs)

    def forward[T: int, K: int](
        self,
        *,
        current: NumPyBicycleObstacleStates[K],
        input: NumPyBicycleObstacleControlInputSequences[T, K],
    ) -> NumPyObstacleStates[T, K]:
        result = simulate(
            input.array,
            current.array,
            time_step_size=self.time_step_size,
            wheelbase=self.wheelbase,
            speed_limits=(float("-inf"), float("inf")),
            steering_limits=(float("-inf"), float("inf")),
            acceleration_limits=(float("-inf"), float("inf")),
        )

        return NumPyObstaclePositionsAndHeadings.create(
            x=result[:, 0, :], y=result[:, 1, :], heading=result[:, 2, :]
        )


def simulate[T: int, M: int](
    inputs: ControlInputBatchArray[T, M],
    initial: StatesAtTimeStep[M],
    *,
    time_step_size: float,
    wheelbase: float,
    speed_limits: tuple[float, float],
    steering_limits: tuple[float, float],
    acceleration_limits: tuple[float, float],
) -> StateBatchArray[T, M]:
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


def estimate_heading_from[K: int](
    history: NumPyBicycleObstacleStatesHistory[int, K], *, time_step_size: float
) -> Array[Dims[K]]:
    assert history.horizon >= 2, (
        "At least two history steps are required to estimate heading."
    )

    delta_x = history.x()[-1] - history.x()[-2]
    delta_y = history.y()[-1] - history.y()[-2]

    headings = np.arctan2(delta_y, delta_x)

    assert shape_of(headings, matches=(history.count,), name="estimated headings")

    return headings


def estimate_speed_from[K: int](
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
