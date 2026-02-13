from typing import Never, Self, overload, cast, Sequence, Final, Any
from dataclasses import dataclass
from functools import cached_property

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
    BicycleD_o,
    BICYCLE_D_O,
    BicyclePoseD_o,
    BICYCLE_POSE_D_O,
    BicyclePositionD_o,
    BICYCLE_POSITION_D_O,
    DynamicalModel,
    ObstacleModel,
    EstimatedObstacleStates,
    CovarianceExtractor,
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

type BicycleObstacleStateCovarianceArray[T: int, K: int] = Array[
    Dims[T, BicycleD_o, BicycleD_o, K]
]
type BicycleObstaclePoseCovarianceArray[T: int, K: int] = Array[
    Dims[T, BicyclePoseD_o, BicyclePoseD_o, K]
]
type BicycleObstaclePositionCovarianceArray[T: int, K: int] = Array[
    Dims[T, BicyclePositionD_o, BicyclePositionD_o, K]
]


@dataclass(frozen=True)
class NumPyBicycleState(BicycleState, NumPyState[BicycleD_x]):
    """Kinematic bicycle state: [x, y, heading, speed]."""

    _array: StateArray

    @staticmethod
    def create(
        *, x: float, y: float, heading: float, speed: float
    ) -> "NumPyBicycleState":
        return NumPyBicycleState(array([x, y, heading, speed], shape=(BICYCLE_D_X,)))

    def __array__(self, dtype: DataType | None = None) -> StateArray:
        return self.array

    @property
    def dimension(self) -> BicycleD_x:
        return self.array.shape[0]

    @property
    def x(self) -> float:
        return self.array[0]

    @property
    def y(self) -> float:
        return self.array[1]

    @property
    def heading(self) -> float:
        return self.array[2]

    @property
    def speed(self) -> float:
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
        return NumPyBicycleStateBatch(array)

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[NumPyBicycleState], *, horizon: T_ | None = None
    ) -> "NumPyBicycleStateBatch[int, D[1]]":
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

    @property
    def horizon(self) -> T:
        return self.batch.horizon

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def rollout_count(self) -> M:
        return self.batch.rollout_count


@dataclass(frozen=True)
class NumPyBicycleControlInputSequence[T: int](
    BicycleControlInputSequence[T], NumPyControlInputSequence[T, BicycleD_u]
):
    """Control inputs: [acceleration, steering_angle]."""

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
        array = np.zeros((horizon, BICYCLE_D_U, rollout_count))

        assert shape_of(array, matches=(horizon, BICYCLE_D_U, rollout_count))

        return NumPyBicycleControlInputBatch(array)

    @staticmethod
    def create[T_: int, M_: int](
        *, array: Array[Dims[T_, BicycleD_u, M_]]
    ) -> "NumPyBicycleControlInputBatch[T_, M_]":

        return NumPyBicycleControlInputBatch(array)

    @staticmethod
    def of[T_: int](
        sequence: NumPyBicycleControlInputSequence[T_],
    ) -> "NumPyBicycleControlInputBatch[T_, D[1]]":
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
    array: Array[Dims[BicycleD_o, K]]

    @staticmethod
    def wrap[K_: int](
        array: Array[Dims[BicycleD_o, K_]],
    ) -> "NumPyBicycleObstacleStates[K_]":
        return NumPyBicycleObstacleStates(array)

    @staticmethod
    def create(
        *,
        x: Array[Dims[K]],
        y: Array[Dims[K]],
        heading: Array[Dims[K]],
        speed: Array[Dims[K]],
    ) -> "NumPyBicycleObstacleStates[K]":
        array = np.stack([x, y, heading, speed], axis=0)

        assert shape_of(array, matches=(BICYCLE_D_O, x.shape[0]))

        return NumPyBicycleObstacleStates(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[BicycleD_o, K]]:
        return self.array

    def heading(self) -> Array[Dims[K]]:
        return self.array[2, :]

    def speed(self) -> Array[Dims[K]]:
        return self.array[3, :]

    @property
    def dimension(self) -> BicycleD_o:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]


@dataclass(frozen=True)
class NumPyBicycleObstacleStateSequences[T: int, K: int]:
    array: Array[Dims[T, BicycleD_o, K]]

    @staticmethod
    def create(
        *,
        x: Array[Dims[T, K]],
        y: Array[Dims[T, K]],
        heading: Array[Dims[T, K]],
        speed: Array[Dims[T, K]],
    ) -> "NumPyBicycleObstacleStateSequences[T, K]":
        T, K = x.shape
        array = np.stack([x, y, heading, speed], axis=1)

        assert shape_of(array, matches=(T, BICYCLE_D_O, K))

        return NumPyBicycleObstacleStateSequences(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_o, K]]:
        return self.array

    def single(self) -> Never:
        # TODO: Fix this!
        raise NotImplementedError(
            "single() is not implemented for NumPyBicycleObstacleStateSequences."
        )

    def x(self) -> Array[Dims[T, K]]:
        return self.array[:, 0, :]

    def y(self) -> Array[Dims[T, K]]:
        return self.array[:, 1, :]

    def heading(self) -> Array[Dims[T, K]]:
        return self.array[:, 2, :]

    def speed(self) -> Array[Dims[T, K]]:
        return self.array[:, 3, :]

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_o:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]


@dataclass(frozen=True)
class NumPyBicycleObstacleInputs[K: int]:
    accelerations: Array[Dims[K]]
    steering_angles: Array[Dims[K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[BicycleD_u, K]]:
        return self._array

    def zeroed(
        self, *, acceleration: bool = False, steering_angle: bool = False
    ) -> "NumPyBicycleObstacleInputs[K]":
        """Returns a version of the inputs with the specified components zeroed out."""
        return NumPyBicycleObstacleInputs(
            accelerations=np.zeros_like(self.accelerations)
            if acceleration
            else self.accelerations,
            steering_angles=np.zeros_like(self.steering_angles)
            if steering_angle
            else self.steering_angles,
        )

    @property
    def dimension(self) -> BicycleD_u:
        return BICYCLE_D_U

    @property
    def count(self) -> K:
        return self.steering_angles.shape[0]

    @cached_property
    def _array(self) -> Array[Dims[BicycleD_u, K]]:
        array = np.stack([self.accelerations, self.steering_angles], axis=0)

        assert shape_of(array, matches=(BICYCLE_D_U, self.count))

        return array


@dataclass(frozen=True)
class NumPyBicycleObstacleControlInputSequences[T: int, K: int]:
    array: Array[Dims[T, BicycleD_u, K]]

    @staticmethod
    def wrap[T_: int, K_: int](
        array: Array[Dims[T_, BicycleD_u, K_]],
    ) -> "NumPyBicycleObstacleControlInputSequences[T_, K_]":
        return NumPyBicycleObstacleControlInputSequences(array)

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

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_u, K]]:
        return self.array

    def steering_angles(self) -> Array[Dims[T, K]]:
        return self.array[:, 1, :]

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_u:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]


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
    """Kinematic bicycle with rear-axle reference, Euler-integrated with configurable limits."""

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
        NumPyBicycleObstacleInputs,
        NumPyBicycleObstacleControlInputSequences,
        NumPyBicycleObstacleStateSequences,
    ]
):
    """Propagates bicycle kinematics forward given states and control inputs."""

    time_step_size: float
    wheelbase: float

    @staticmethod
    def create(
        *, time_step_size: float, wheelbase: float = 1.0
    ) -> "NumPyBicycleObstacleModel":
        return NumPyBicycleObstacleModel(
            time_step_size=time_step_size,
            wheelbase=wheelbase,
        )

    def input_to_maintain[K: int](
        self,
        inputs: NumPyBicycleObstacleInputs[K],
        *,
        states: NumPyBicycleObstacleStates[K],
        horizon: int,
    ) -> NumPyBicycleObstacleControlInputSequences[int, K]:
        return NumPyBicycleObstacleControlInputSequences.create(
            accelerations=np.tile(inputs.accelerations[np.newaxis, :], (horizon, 1)),
            steering_angles=np.tile(
                inputs.steering_angles[np.newaxis, :], (horizon, 1)
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

    def state_jacobian[T: int, K: int](
        self,
        *,
        states: NumPyBicycleObstacleStateSequences[T, K],
        inputs: NumPyBicycleObstacleControlInputSequences[T, K],
    ) -> Array[Dims[T, BicycleD_o, BicycleD_o, K]]:
        return state_jacobian(
            heading=states.heading(),
            speed=states.speed(),
            steering_angle=inputs.steering_angles(),
            time_step_size=self.time_step_size,
            wheelbase=self.wheelbase,
        )

    def input_jacobian[T: int, K: int](
        self,
        *,
        states: NumPyBicycleObstacleStateSequences[T, K],
        inputs: NumPyBicycleObstacleControlInputSequences[T, K],
    ) -> Array[Dims[T, BicycleD_o, BicycleD_u, K]]:
        return input_jacobian(
            speed=states.speed(),
            steering_angle=inputs.steering_angles(),
            time_step_size=self.time_step_size,
            wheelbase=self.wheelbase,
        )


class NumPyBicyclePoseCovarianceExtractor(CovarianceExtractor):
    """Extracts the pose-related state covariance from the full state covariance for bicycle obstacles."""

    def __call__[T: int, K: int](
        self, covariance: BicycleObstacleStateCovarianceArray[T, K]
    ) -> BicycleObstaclePoseCovarianceArray[T, K]:
        return covariance[:, :BICYCLE_POSE_D_O, :BICYCLE_POSE_D_O, :]


class NumPyBicyclePositionCovarianceExtractor(CovarianceExtractor):
    """Extracts x and y covariance from the full state covariance for bicycle obstacles."""

    def __call__[T: int, K: int](
        self, covariance: BicycleObstacleStateCovarianceArray[T, K]
    ) -> BicycleObstaclePositionCovarianceArray[T, K]:
        return covariance[:, :BICYCLE_POSITION_D_O, :BICYCLE_POSITION_D_O, :]


@dataclass(frozen=True)
class NumPyFiniteDifferenceBicycleStateEstimator:
    time_step_size: float
    wheelbase: float

    @staticmethod
    def create(
        *, time_step_size: float, wheelbase: float
    ) -> "NumPyFiniteDifferenceBicycleStateEstimator":
        return NumPyFiniteDifferenceBicycleStateEstimator(
            time_step_size=time_step_size, wheelbase=wheelbase
        )

    def estimate_from[K: int](
        self, history: NumPyBicycleObstacleStatesHistory[int, K]
    ) -> EstimatedObstacleStates[
        NumPyBicycleObstacleStates[K], NumPyBicycleObstacleInputs[K]
    ]:
        speeds = self.estimate_speeds_from(history)
        accelerations = self.estimate_accelerations_from(history, speeds=speeds)
        steering_angles = self.estimate_steering_angles_from(history, speeds=speeds)

        return EstimatedObstacleStates(
            states=NumPyBicycleObstacleStates.create(
                x=history.x()[-1],
                y=history.y()[-1],
                heading=history.heading()[-1, :],
                speed=speeds,
            ),
            inputs=NumPyBicycleObstacleInputs(
                accelerations=accelerations, steering_angles=steering_angles
            ),
        )

    def estimate_speeds_from[K: int](
        self, history: NumPyBicycleObstacleStatesHistory[int, K]
    ) -> Array[Dims[K]]:
        """Estimates speeds from position history using finite differences.

        Speed is computed as the projection of the displacement onto the heading direction,
        which can be negative for reverse motion:

        # TODO: Fix KaTeX formatting.
        $$v_t = ((x_t - x_{t-1}) * cos(\\theta_t) + (y_t - y_{t-1}) * sin(\\theta_t)) / (\\Delta t)$$
        """
        if history.horizon < 2:
            return cast(Array[Dims[K]], np.zeros((history.count,)))

        x = history.x()
        y = history.y()
        heading = history.heading()

        return self._estimate_speeds_from(
            x_current=x[-1],
            y_current=y[-1],
            x_previous=x[-2],
            y_previous=y[-2],
            heading_current=heading[-1],
        )

    def estimate_accelerations_from[K: int](
        self,
        history: NumPyBicycleObstacleStatesHistory[int, K],
        *,
        speeds: Array[Dims[K]],
    ) -> Array[Dims[K]]:
        """Estimates accelerations from speed history using finite differences.

        Acceleration is computed as the change in speed over time:

        $$a_t = (v_t - v_{t-1}) / (\\Delta t)$$
        """
        if history.horizon < 3:
            return cast(Array[Dims[K]], np.zeros((history.count,)))

        x = history.x()
        y = history.y()
        heading = history.heading()

        return self._estimate_accelerations_from(
            speeds_current=speeds,
            speeds_previous=self._estimate_speeds_from(
                x_current=x[-2],
                y_current=y[-2],
                x_previous=x[-3],
                y_previous=y[-3],
                heading_current=heading[-2],
            ),
        )

    def estimate_steering_angles_from[K: int](
        self,
        history: NumPyBicycleObstacleStatesHistory[int, K],
        *,
        speeds: Array[Dims[K]],
    ) -> Array[Dims[K]]:
        """Estimates steering angles from heading change and speed.

        From the kinematic bicycle model: δ = arctan(L * ω / v)
        where ω = dθ/dt is the heading rate.

        When speed is very small, the steering angle estimate is unreliable,
        so we return zero in those cases.
        """
        if history.horizon < 2:
            return cast(Array[Dims[K]], np.zeros((history.count,)))

        heading = history.heading()

        return self._estimate_steering_angles_from(
            heading_current=heading[-1], heading_previous=heading[-2], speeds=speeds
        )

    def _estimate_speeds_from[K: int](
        self,
        *,
        x_current: Array[Dims[K]],
        y_current: Array[Dims[K]],
        x_previous: Array[Dims[K]],
        y_previous: Array[Dims[K]],
        heading_current: Array[Dims[K]],
    ) -> Array[Dims[K]]:
        delta_x = x_current - x_previous
        delta_y = y_current - y_previous

        speeds = (
            delta_x * np.cos(heading_current) + delta_y * np.sin(heading_current)
        ) / self.time_step_size

        assert shape_of(speeds, matches=(x_current.shape[0],), name="estimated speeds")

        return speeds

    def _estimate_accelerations_from[K: int](
        self, *, speeds_current: Array[Dims[K]], speeds_previous: Array[Dims[K]]
    ) -> Array[Dims[K]]:
        accelerations = (speeds_current - speeds_previous) / self.time_step_size

        assert shape_of(
            accelerations,
            matches=(speeds_current.shape[0],),
            name="estimated accelerations",
        )

        return accelerations

    def _estimate_steering_angles_from[K: int](
        self,
        *,
        heading_current: Array[Dims[K]],
        heading_previous: Array[Dims[K]],
        speeds: Array[Dims[K]],
    ) -> Array[Dims[K]]:
        heading_velocity = (heading_current - heading_previous) / self.time_step_size

        with np.errstate(invalid="ignore"):
            steering_angles = np.where(
                np.abs(speeds) > 1e-6,
                np.arctan(heading_velocity * self.wheelbase / speeds),
                0.0,
            )

        assert shape_of(
            steering_angles,
            matches=(heading_current.shape[0],),
            name="estimated steering angles",
        )

        return steering_angles


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


def state_jacobian[T: int, K: int](
    heading: Array[Dims[T, K]],
    speed: Array[Dims[T, K]],
    steering_angle: Array[Dims[T, K]],
    *,
    time_step_size: float,
    wheelbase: float,
) -> Array[Dims[T, BicycleD_o, BicycleD_o, K]]:
    """Computes the state Jacobian F = ∂f/∂x for the bicycle model."""
    v, theta, delta = speed, heading, steering_angle

    T, K = heading.shape
    F = np.zeros((T, BICYCLE_D_O, BICYCLE_D_O, K))

    dt = time_step_size

    F[:, 0, 0, :] = 1.0
    F[:, 1, 1, :] = 1.0
    F[:, 2, 2, :] = 1.0
    F[:, 3, 3, :] = 1.0

    F[:, 0, 2, :] = -v * np.sin(theta) * dt
    F[:, 0, 3, :] = np.cos(theta) * dt
    F[:, 1, 2, :] = v * np.cos(theta) * dt
    F[:, 1, 3, :] = np.sin(theta) * dt
    F[:, 2, 3, :] = np.tan(delta) / wheelbase * dt

    assert shape_of(F, matches=(T, BICYCLE_D_O, BICYCLE_D_O, K), name="state_jacobian")

    return F


def input_jacobian[T: int, K: int](
    speed: Array[Dims[T, K]],
    steering_angle: Array[Dims[T, K]],
    *,
    time_step_size: float,
    wheelbase: float,
) -> Array[Dims[T, BicycleD_o, BicycleD_u, K]]:
    """Computes the input Jacobian G = ∂f/∂u for the bicycle model.

    For bicycle model: u = [acceleration, steering_angle]
    G describes how control input uncertainty enters the state dynamics.
    """
    v, delta = speed, steering_angle

    T, K = speed.shape
    G = np.zeros((T, BICYCLE_D_O, BICYCLE_D_U, K))

    dt = time_step_size

    # ∂θ/∂δ = v / (L * cos²(δ)) * dt
    G[:, 2, 1, :] = v / (wheelbase * np.cos(delta) ** 2) * dt

    # ∂v/∂a = dt
    G[:, 3, 0, :] = dt

    assert shape_of(G, matches=(T, BICYCLE_D_O, BICYCLE_D_U, K), name="input_jacobian")

    return G
