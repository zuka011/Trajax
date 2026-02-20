from typing import Self, overload, cast, Sequence, Final, Any
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    DataType,
    NumPyState,
    NumPyStateSequence,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyUnicycleObstacleStatesHistory,
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
    UnicycleD_o,
    UNICYCLE_D_O,
    PoseD_o,
    DynamicalModel,
    ObstacleModel,
    ObstacleStateEstimator,
    EstimatedObstacleStates,
)
from faran.filters import (
    NumPyExtendedKalmanFilter,
    NumPyGaussianBelief,
    NumPyNoiseCovarianceDescription,
    NumPyUnscentedKalmanFilter,
    numpy_kalman_filter,
)
from faran.obstacles import NumPyObstacle2dPoses
from faran.models.common import SMALL_UNCERTAINTY, LARGE_UNCERTAINTY
from faran.models.basic import invalid_obstacle_filter_from

from numtypes import Array, Dims, D, shape_of, array

import numpy as np

NO_LIMITS: Final = (float("-inf"), float("inf"))

UNICYCLE_ESTIMATION_D_X: Final = 5
UNICYCLE_OBSERVATION_D_O: Final = 3

type UnicycleEstimationD_x = D[5]
type UnicycleObservationD_o = D[3]

type StateArray = Array[Dims[UnicycleD_x]]
type ControlInputSequenceArray[T: int] = Array[Dims[T, UnicycleD_u]]
type StateBatchArray[T: int, M: int] = Array[Dims[T, UnicycleD_x, M]]
type ControlInputBatchArray[T: int, M: int] = Array[Dims[T, UnicycleD_u, M]]

type StatesAtTimeStep[M: int] = Array[Dims[UnicycleD_x, M]]
type ControlInputsAtTimeStep[M: int] = Array[Dims[UnicycleD_u, M]]

type EstimationStateCovarianceArray = Array[
    Dims[UnicycleEstimationD_x, UnicycleEstimationD_x]
]
type ProcessNoiseCovarianceArray = Array[
    Dims[UnicycleEstimationD_x, UnicycleEstimationD_x]
]
type ObservationNoiseCovarianceArray = Array[
    Dims[UnicycleObservationD_o, UnicycleObservationD_o]
]
type ObservationMatrix = Array[Dims[UnicycleObservationD_o, UnicycleEstimationD_x]]

type UnicycleGaussianBelief[K: int] = NumPyGaussianBelief[UnicycleEstimationD_x, K]
type NumPyUnicycleObstacleCovariances[K: int] = Array[
    Dims[UnicycleEstimationD_x, UnicycleEstimationD_x, K]
]
type KalmanFilter = NumPyExtendedKalmanFilter | NumPyUnscentedKalmanFilter


@dataclass(frozen=True)
class NumPyUnicycleState(UnicycleState, NumPyState[UnicycleD_x]):
    """Kinematic unicycle state: [x, y, heading]."""

    _array: StateArray

    @staticmethod
    def create(*, x: float, y: float, heading: float) -> "NumPyUnicycleState":
        """Creates a NumPy unicycle state from individual state components."""
        return NumPyUnicycleState(array([x, y, heading], shape=(UNICYCLE_D_X,)))

    def __array__(self, dtype: DataType | None = None) -> StateArray:
        return self.array

    @property
    def dimension(self) -> UnicycleD_x:
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
    def array(self) -> StateArray:
        return self._array


@dataclass(kw_only=True, frozen=True)
class NumPyUnicycleStateSequence[T: int, M: int = Any](
    UnicycleStateSequence, NumPyStateSequence[T, UnicycleD_x]
):
    batch: "NumPyUnicycleStateBatch[T, M]"
    rollout: int

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[NumPyUnicycleState], *, horizon: T_ | None = None
    ) -> "NumPyUnicycleStateSequence[T_, D[1]]":
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        assert shape_of(array, matches=(horizon, UNICYCLE_D_X, 1))

        return NumPyUnicycleStateSequence(
            batch=NumPyUnicycleStateBatch.wrap(array), rollout=0
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, UnicycleD_x]]:
        return self.array

    def step(self, index: int) -> NumPyUnicycleState:
        return NumPyUnicycleState(self.array[index])

    def batched(self) -> "NumPyUnicycleStateBatch[T, D[1]]":
        return NumPyUnicycleStateBatch.wrap(self.array[..., np.newaxis])

    def x(self) -> Array[Dims[T]]:
        return self.array[:, 0]

    def y(self) -> Array[Dims[T]]:
        return self.array[:, 1]

    def heading(self) -> Array[Dims[T]]:
        return self.array[:, 2]

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> UnicycleD_x:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[T, UnicycleD_x]]:
        return self.batch.array[:, :, self.rollout]


@dataclass(frozen=True)
class NumPyUnicycleStateBatch[T: int, M: int](
    UnicycleStateBatch[T, M], NumPyStateBatch[T, UnicycleD_x, M]
):
    _array: StateBatchArray[T, M]

    @staticmethod
    def wrap[T_: int, M_: int](
        array: StateBatchArray[T_, M_],
    ) -> "NumPyUnicycleStateBatch[T_, M_]":
        return NumPyUnicycleStateBatch(array)

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[NumPyUnicycleState], *, horizon: T_ | None = None
    ) -> "NumPyUnicycleStateBatch[int, D[1]]":
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        assert shape_of(array, matches=(horizon, UNICYCLE_D_X, 1))

        return NumPyUnicycleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> StateBatchArray[T, M]:
        return self.array

    def heading(self) -> Array[Dims[T, M]]:
        return self.array[:, 2, :]

    def rollout(self, index: int) -> NumPyUnicycleStateSequence[T, M]:
        return NumPyUnicycleStateSequence(batch=self, rollout=index)

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> UnicycleD_x:
        return self.array.shape[1]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]

    @property
    def positions(self) -> "NumPyUnicyclePositions[T, M]":
        return NumPyUnicyclePositions(batch=self)

    @property
    def array(self) -> StateBatchArray[T, M]:
        return self._array


@dataclass(frozen=True)
class NumPyUnicyclePositions[T: int, M: int](UnicyclePositions[T, M]):
    batch: NumPyUnicycleStateBatch[T, M]

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
class NumPyUnicycleControlInputSequence[T: int](
    UnicycleControlInputSequence[T], NumPyControlInputSequence[T, UnicycleD_u]
):
    """Control inputs: [linear velocity, angular velocity]."""

    _array: ControlInputSequenceArray[T]

    @staticmethod
    def zeroes[T_: int](horizon: T_) -> "NumPyUnicycleControlInputSequence[T_]":
        """Creates a zeroed control input sequence for the given horizon."""
        array = np.zeros((horizon, UNICYCLE_D_U))

        assert shape_of(array, matches=(horizon, UNICYCLE_D_U))

        return NumPyUnicycleControlInputSequence(array)

    def __array__(self, dtype: DataType | None = None) -> ControlInputSequenceArray[T]:
        return self.array

    @overload
    def similar(self, *, array: Array[Dims[T, UnicycleD_u]]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Array[Dims[L, UnicycleD_u]], length: L
    ) -> "NumPyUnicycleControlInputSequence[L]": ...

    def similar[L: int](
        self, *, array: Array[Dims[L, UnicycleD_u]], length: L | None = None
    ) -> "Self | NumPyUnicycleControlInputSequence[L]":
        # NOTE: "Wrong" cast to silence the type checker.
        effective_length = cast(T, length if length is not None else array.shape[0])

        assert shape_of(
            array, matches=(effective_length, self.dimension), name="similar array"
        )

        return self.__class__(array)

    def linear_velocities(self) -> Array[Dims[T]]:
        return self.array[:, 0]

    def angular_velocities(self) -> Array[Dims[T]]:
        return self.array[:, 1]

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> UnicycleD_u:
        return self.array.shape[1]

    @property
    def array(self) -> ControlInputSequenceArray[T]:
        return self._array


@dataclass(frozen=True)
class NumPyUnicycleControlInputBatch[T: int, M: int](
    UnicycleControlInputBatch[T, M], NumPyControlInputBatch[T, UnicycleD_u, M]
):
    _array: ControlInputBatchArray[T, M]

    @staticmethod
    def zero[T_: int, M_: int](
        *, horizon: T_, rollout_count: M_ = 1
    ) -> "NumPyUnicycleControlInputBatch[T_, M_]":
        array = np.zeros((horizon, UNICYCLE_D_U, rollout_count))

        assert shape_of(array, matches=(horizon, UNICYCLE_D_U, rollout_count))

        return NumPyUnicycleControlInputBatch(array)

    @staticmethod
    def create[T_: int, M_: int](
        *, array: Array[Dims[T_, UnicycleD_u, M_]]
    ) -> "NumPyUnicycleControlInputBatch[T_, M_]":

        return NumPyUnicycleControlInputBatch(array)

    @staticmethod
    def of[T_: int](
        sequence: NumPyUnicycleControlInputSequence[T_],
    ) -> "NumPyUnicycleControlInputBatch[T_, D[1]]":
        array = sequence.array[..., np.newaxis]

        assert shape_of(array, matches=(sequence.horizon, UNICYCLE_D_U, 1))

        return NumPyUnicycleControlInputBatch(array)

    def __array__(self, dtype: DataType | None = None) -> ControlInputBatchArray[T, M]:
        return self.array

    def linear_velocity(self) -> Array[Dims[T, M]]:
        return self.array[:, 0, :]

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> UnicycleD_u:
        return self.array.shape[1]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]

    @property
    def array(self) -> ControlInputBatchArray[T, M]:
        return self._array


@dataclass(frozen=True)
class NumPyUnicycleObstacleStates[K: int]:
    _array: Array[Dims[UnicycleD_o, K]]

    @staticmethod
    def wrap[K_: int](
        array: Array[Dims[UnicycleD_o, K_]],
    ) -> "NumPyUnicycleObstacleStates[K_]":
        return NumPyUnicycleObstacleStates(array)

    @staticmethod
    def create(
        *,
        x: Array[Dims[K]],
        y: Array[Dims[K]],
        heading: Array[Dims[K]],
    ) -> "NumPyUnicycleObstacleStates[K]":
        array = np.stack([x, y, heading], axis=0)

        assert shape_of(array, matches=(UNICYCLE_D_O, x.shape[0]))

        return NumPyUnicycleObstacleStates(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[UnicycleD_o, K]]:
        return self.array

    def x(self) -> Array[Dims[K]]:
        return self.array[0, :]

    def y(self) -> Array[Dims[K]]:
        return self.array[1, :]

    def heading(self) -> Array[Dims[K]]:
        return self.array[2, :]

    @property
    def dimension(self) -> UnicycleD_o:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[UnicycleD_o, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyUnicycleObstacleStateSequences[T: int, K: int]:
    _array: Array[Dims[T, UnicycleEstimationD_x, K]]
    _covariance: Array[Dims[T, UnicycleEstimationD_x, UnicycleEstimationD_x, K]]

    @staticmethod
    def create[K_: int, T_: int = int](
        predictions: Sequence[UnicycleGaussianBelief[K_]],
    ) -> "NumPyUnicycleObstacleStateSequences[T_, K_]":
        assert len(predictions) > 0, "Predictions sequence must not be empty."

        T = cast(T_, len(predictions))
        K = predictions[0].mean.shape[1]

        array = np.stack([belief.mean for belief in predictions], axis=0)
        covariance = np.stack([belief.covariance for belief in predictions], axis=0)

        assert shape_of(array, matches=(T, UNICYCLE_ESTIMATION_D_X, K))
        assert shape_of(
            covariance, matches=(T, UNICYCLE_ESTIMATION_D_X, UNICYCLE_ESTIMATION_D_X, K)
        )

        return NumPyUnicycleObstacleStateSequences(array, covariance)

    def __array__(
        self, dtype: DataType | None = None
    ) -> Array[Dims[T, UnicycleEstimationD_x, K]]:
        return self.array

    def x(self) -> Array[Dims[T, K]]:
        return self.array[:, 0, :]

    def y(self) -> Array[Dims[T, K]]:
        return self.array[:, 1, :]

    def heading(self) -> Array[Dims[T, K]]:
        return self.array[:, 2, :]

    def covariance(
        self,
    ) -> Array[Dims[T, UnicycleEstimationD_x, UnicycleEstimationD_x, K]]:
        return self._covariance

    def pose_covariance(self) -> Array[Dims[T, PoseD_o, PoseD_o, K]]:
        return self._covariance[:, :3, :3, :]

    def pose(self) -> NumPyObstacle2dPoses[T, K]:
        return NumPyObstacle2dPoses.create(
            x=self.x(),
            y=self.y(),
            heading=self.heading(),
            covariance=self.pose_covariance(),
        )

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> UnicycleEstimationD_x:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, UnicycleEstimationD_x, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyUnicycleObstacleInputs[K: int]:
    _linear_velocities: Array[Dims[K]]
    _angular_velocities: Array[Dims[K]]

    @staticmethod
    def wrap[K_: int](
        array: Array[Dims[UnicycleD_u, K_]],
    ) -> "NumPyUnicycleObstacleInputs[K_]":
        linear, angular = array
        return NumPyUnicycleObstacleInputs(
            _linear_velocities=linear, _angular_velocities=angular
        )

    @staticmethod
    def create(
        *,
        linear_velocities: Array[Dims[K]],
        angular_velocities: Array[Dims[K]],
    ) -> "NumPyUnicycleObstacleInputs[K]":
        return NumPyUnicycleObstacleInputs(
            _linear_velocities=linear_velocities, _angular_velocities=angular_velocities
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[UnicycleD_u, K]]:
        return self._array

    def linear_velocities(self) -> Array[Dims[K]]:
        return self._linear_velocities

    def angular_velocities(self) -> Array[Dims[K]]:
        return self._angular_velocities

    def zeroed(
        self, *, linear_velocity: bool = False, angular_velocity: bool = False
    ) -> "NumPyUnicycleObstacleInputs[K]":
        """Returns a version of the inputs with the specified components zeroed out."""
        return NumPyUnicycleObstacleInputs(
            _linear_velocities=np.zeros_like(self._linear_velocities)
            if linear_velocity
            else self._linear_velocities,
            _angular_velocities=np.zeros_like(self._angular_velocities)
            if angular_velocity
            else self._angular_velocities,
        )

    @property
    def dimension(self) -> UnicycleD_u:
        return UNICYCLE_D_U

    @property
    def count(self) -> K:
        return self._linear_velocities.shape[0]

    @property
    def array(self) -> Array[Dims[UnicycleD_u, K]]:
        return self._array

    @cached_property
    def _array(self) -> Array[Dims[UnicycleD_u, K]]:
        return np.stack([self._linear_velocities, self._angular_velocities], axis=0)


@dataclass(kw_only=True, frozen=True)
class NumPyUnicycleModel(
    DynamicalModel[
        NumPyUnicycleState,
        NumPyUnicycleStateSequence,
        NumPyUnicycleStateBatch,
        NumPyUnicycleControlInputSequence,
        NumPyUnicycleControlInputBatch,
    ],
):
    """Kinematic unicycle with direct velocity control, Euler-integrated with configurable limits."""

    _time_step_size: float
    speed_limits: tuple[float, float]
    angular_velocity_limits: tuple[float, float]

    @staticmethod
    def create(
        *,
        time_step_size: float,
        speed_limits: tuple[float, float] | None = None,
        angular_velocity_limits: tuple[float, float] | None = None,
    ) -> "NumPyUnicycleModel":

        return NumPyUnicycleModel(
            _time_step_size=time_step_size,
            speed_limits=speed_limits if speed_limits is not None else NO_LIMITS,
            angular_velocity_limits=angular_velocity_limits
            if angular_velocity_limits is not None
            else NO_LIMITS,
        )

    def simulate[T: int, M: int](
        self,
        inputs: NumPyUnicycleControlInputBatch[T, M],
        initial_state: NumPyUnicycleState,
    ) -> NumPyUnicycleStateBatch[T, M]:
        rollout_count = inputs.rollout_count

        initial = np.stack(
            [
                np.full(rollout_count, initial_state.x),
                np.full(rollout_count, initial_state.y),
                np.full(rollout_count, initial_state.heading),
            ]
        )

        return NumPyUnicycleStateBatch(
            simulate(
                inputs.array,
                initial,
                time_step_size=self.time_step_size,
                speed_limits=self.speed_limits,
                angular_velocity_limits=self.angular_velocity_limits,
            )
        )

    def step[T: int](
        self, inputs: NumPyUnicycleControlInputSequence[T], state: NumPyUnicycleState
    ) -> NumPyUnicycleState:
        state_as_rollouts = state.array.reshape(-1, 1)
        first_input = inputs.array[0].reshape(-1, 1)

        assert shape_of(
            state_as_rollouts, matches=(UNICYCLE_D_X, 1), name="state reshaped for step"
        )
        assert shape_of(
            first_input,
            matches=(UNICYCLE_D_U, 1),
            name="first control input reshaped for step",
        )

        return NumPyUnicycleState(
            step(
                state_as_rollouts,
                first_input,
                time_step_size=self.time_step_size,
                speed_limits=self.speed_limits,
                angular_velocity_limits=self.angular_velocity_limits,
            )[:, 0]
        )

    def forward[T: int](
        self, inputs: NumPyUnicycleControlInputSequence[T], state: NumPyUnicycleState
    ) -> NumPyUnicycleStateSequence[T]:
        return self.simulate(NumPyUnicycleControlInputBatch.of(inputs), state).rollout(
            0
        )

    @property
    def time_step_size(self) -> float:
        return self._time_step_size


@dataclass(frozen=True)
class NumPyUnicycleStateEstimationModel:
    """Kinematic unicycle model used for obstacle state estimation."""

    time_step_size: float
    initial_state_covariance: EstimationStateCovarianceArray

    @staticmethod
    def create(
        *,
        time_step_size: float,
        initial_state_covariance: EstimationStateCovarianceArray | None = None,
    ) -> "NumPyUnicycleStateEstimationModel":
        if initial_state_covariance is None:
            initial_state_covariance = (
                NumPyUnicycleStateEstimationModel.default_initial_state_covariance()
            )

        return NumPyUnicycleStateEstimationModel(
            time_step_size=time_step_size,
            initial_state_covariance=initial_state_covariance,
        )

    @staticmethod
    def default_initial_state_covariance() -> EstimationStateCovarianceArray:
        # NOTE: Sure of pose, unsure of velocities.
        return cast(
            EstimationStateCovarianceArray,
            np.diag(
                [
                    SMALL_UNCERTAINTY,
                    SMALL_UNCERTAINTY,
                    SMALL_UNCERTAINTY,
                    LARGE_UNCERTAINTY,
                    LARGE_UNCERTAINTY,
                ]
            ),
        )

    def __call__[D_x: int, K: int](
        self, state: Array[Dims[D_x, K]]
    ) -> Array[Dims[D_x, K]]:
        dt = self.time_step_size
        x, y, theta, v, omega = state

        return np.array(
            [
                x + v * np.cos(theta) * dt,
                y + v * np.sin(theta) * dt,
                theta + omega * dt,
                v,
                omega,
            ]
        )

    def jacobian[D_x: int, K: int](
        self, state: Array[Dims[D_x, K]]
    ) -> Array[Dims[D_x, D_x, K]]:
        D_x, K = state.shape
        dt = self.time_step_size
        x, y, theta, v, omega = state

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        jacobian = np.zeros((D_x, D_x, K))

        # Partial derivatives for x
        jacobian[0, 0, :] = 1  # ∂x_next/∂x
        jacobian[0, 2, :] = -v * sin_theta * dt  # ∂x_next/∂theta
        jacobian[0, 3, :] = cos_theta * dt  # ∂x_next/∂v

        # Partial derivatives for y
        jacobian[1, 1, :] = 1  # ∂y_next/∂y
        jacobian[1, 2, :] = v * cos_theta * dt  # ∂y_next/∂theta
        jacobian[1, 3, :] = sin_theta * dt  # ∂y_next/∂v

        # Partial derivatives for theta
        jacobian[2, 2, :] = 1  # ∂theta_next/∂theta
        jacobian[2, 4, :] = dt  # ∂theta_next/∂omega

        # Partial derivatives for v and omega are identity since they are assumed constant
        jacobian[3, 3, :] = 1  # ∂v_next/∂v
        jacobian[4, 4, :] = 1  # ∂omega_next/∂omega

        assert shape_of(jacobian, matches=(D_x, D_x, K), name="jacobian")

        return jacobian

    def observations_from[K: int, T: int = int](
        self, history: NumPyUnicycleObstacleStatesHistory[T, K]
    ) -> Array[Dims[T, UnicycleObservationD_o, K]]:
        return np.stack([history.x(), history.y(), history.heading()], axis=1)

    def states_from[K: int](
        self, belief: UnicycleGaussianBelief[K]
    ) -> NumPyUnicycleObstacleStates[K]:
        return NumPyUnicycleObstacleStates.wrap(belief.mean[:3, :])

    def inputs_from[K: int](
        self, belief: UnicycleGaussianBelief[K]
    ) -> NumPyUnicycleObstacleInputs[K]:
        return NumPyUnicycleObstacleInputs.wrap(belief.mean[3:, :])

    def initial_belief_from[K: int](
        self,
        *,
        states: NumPyUnicycleObstacleStates[K],
        inputs: NumPyUnicycleObstacleInputs[K],
        covariances: NumPyUnicycleObstacleCovariances[K] | None = None,
    ) -> UnicycleGaussianBelief:
        augmented = np.concatenate([states.array, inputs.array], axis=0)

        if covariances is None:
            # NOTE: No covariance means we are "certain" about the states.
            covariances = np.broadcast_to(  # type: ignore
                np.eye(UNICYCLE_ESTIMATION_D_X)[:, :, np.newaxis] * SMALL_UNCERTAINTY,
                (UNICYCLE_ESTIMATION_D_X, UNICYCLE_ESTIMATION_D_X, states.count),
            ).copy()

        assert shape_of(
            augmented,
            matches=(UNICYCLE_ESTIMATION_D_X, states.count),
            name="augmented state",
        )
        assert shape_of(
            covariances,
            matches=(UNICYCLE_ESTIMATION_D_X, UNICYCLE_ESTIMATION_D_X, states.count),
            name="initial covariance",
        )

        return NumPyGaussianBelief(mean=augmented, covariance=covariances)

    @cached_property
    def observation_matrix(self) -> ObservationMatrix:
        # NOTE: We observe the pose directly.
        return np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ]
        )


@dataclass(kw_only=True, frozen=True)
class NumPyUnicycleObstacleModel(
    ObstacleModel[
        NumPyUnicycleObstacleStatesHistory,
        NumPyUnicycleObstacleStates,
        NumPyUnicycleObstacleInputs,
        NumPyUnicycleObstacleCovariances,
        NumPyUnicycleObstacleStateSequences,
    ]
):
    """Propagates unicycle kinematics forward given states and control inputs."""

    model: NumPyUnicycleStateEstimationModel
    process_noise_covariance: ProcessNoiseCovarianceArray
    predictor: KalmanFilter

    @staticmethod
    def unscented(
        *,
        time_step_size: float,
        process_noise_covariance: NumPyNoiseCovarianceDescription = 1e-3,
        sigma_point_spread: float = 1.0,
        prior_knowledge: float = 2.0,
    ) -> "NumPyUnicycleObstacleModel":
        """Creates a unicycle obstacle model for obstacle state prediction. This
        model uses the unscented transform for propagating state information through
        the nonlinear unicycle dynamics.

        Args:
            time_step_size: Time step size for state propagation.
            process_noise_covariance: The process noise covariance, either as a
                full covariance array, a diagonal covariance vector, or a scalar
                variance representing isotropic noise.
            sigma_point_spread: Spread of sigma points (α) for the unscented
                transform, controlling how far the sigma points are from the mean.
            prior_knowledge: Prior knowledge (β) about the state distribution
                for the unscented transform.
        """
        return NumPyUnicycleObstacleModel(
            model=NumPyUnicycleStateEstimationModel.create(
                time_step_size=time_step_size
            ),
            process_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                process_noise_covariance, dimension=UNICYCLE_ESTIMATION_D_X
            ),
            predictor=NumPyUnscentedKalmanFilter.create(
                alpha=sigma_point_spread, beta=prior_knowledge
            ),
        )

    def forward[T: int, K: int](
        self,
        *,
        states: NumPyUnicycleObstacleStates[K],
        inputs: NumPyUnicycleObstacleInputs[K],
        covariances: NumPyUnicycleObstacleCovariances[K] | None,
        horizon: T,
    ) -> NumPyUnicycleObstacleStateSequences[T, K]:
        beliefs = []
        last = self.model.initial_belief_from(
            states=states, inputs=inputs, covariances=covariances
        )

        for _ in range(horizon):
            beliefs.append(
                last := self.predictor.predict(
                    belief=last,  # type: ignore
                    state_transition=self.model,
                    process_noise_covariance=self.process_noise_covariance,
                )
            )

        return NumPyUnicycleObstacleStateSequences.create(beliefs)


@dataclass(frozen=True)
class NumPyFiniteDifferenceUnicycleStateEstimator(
    ObstacleStateEstimator[
        NumPyUnicycleObstacleStatesHistory,
        NumPyUnicycleObstacleStates,
        NumPyUnicycleObstacleInputs,
    ]
):
    time_step_size: float

    @staticmethod
    def create(
        *, time_step_size: float
    ) -> "NumPyFiniteDifferenceUnicycleStateEstimator":
        return NumPyFiniteDifferenceUnicycleStateEstimator(
            time_step_size=time_step_size
        )

    def estimate_from[K: int](
        self, history: NumPyUnicycleObstacleStatesHistory[int, K]
    ) -> EstimatedObstacleStates[
        NumPyUnicycleObstacleStates[K], NumPyUnicycleObstacleInputs[K], None
    ]:
        """Estimates current states and inputs from position/heading history using finite differences.

        Computes the following quantities from the unicycle model (requires T ≥ 2, otherwise returns zeros):

        **Linear velocity**:
            Projection of displacement onto the heading direction (negative for reverse):
            $$v_t = \\frac{(x_t - x_{t-1}) \\cos(\\theta_t) + (y_t - y_{t-1}) \\sin(\\theta_t)}{\\Delta t}$$

        **Angular velocity**:
            Change in heading over time:
            $$\\omega_t = \\frac{\\theta_t - \\theta_{t-1}}{\\Delta t}$$

        Args:
            history: History of observed poses with at least one entry.
        """
        assert history.horizon > 0, "History must contain at least one state."

        filter_invalid = invalid_obstacle_filter_from(history, check_recent=2)
        linear_velocities = self.estimate_speeds_from(history)
        angular_velocities = self.estimate_angular_velocities_from(history)

        return EstimatedObstacleStates(
            states=NumPyUnicycleObstacleStates.create(
                x=history.x()[-1],
                y=history.y()[-1],
                heading=history.heading()[-1],
            ),
            inputs=NumPyUnicycleObstacleInputs.create(
                linear_velocities=filter_invalid(linear_velocities),
                angular_velocities=filter_invalid(angular_velocities),
            ),
            covariance=None,
        )

    def estimate_speeds_from[K: int](
        self, history: NumPyUnicycleObstacleStatesHistory[int, K]
    ) -> Array[Dims[K]]:
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

    def estimate_angular_velocities_from[K: int](
        self, history: NumPyUnicycleObstacleStatesHistory[int, K]
    ) -> Array[Dims[K]]:
        if history.horizon < 2:
            return cast(Array[Dims[K]], np.zeros((history.count,)))

        heading = history.heading()

        return self._estimate_angular_velocities_from(
            heading_current=heading[-1],
            heading_previous=heading[-2],
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

    def _estimate_angular_velocities_from[K: int](
        self,
        *,
        heading_current: Array[Dims[K]],
        heading_previous: Array[Dims[K]],
    ) -> Array[Dims[K]]:
        angular_velocities = (heading_current - heading_previous) / self.time_step_size

        assert shape_of(
            angular_velocities,
            matches=(heading_current.shape[0],),
            name="estimated angular velocities",
        )

        return angular_velocities


@dataclass(frozen=True)
class NumPyKfUnicycleStateEstimator(
    ObstacleStateEstimator[
        NumPyUnicycleObstacleStatesHistory,
        NumPyUnicycleObstacleStates,
        NumPyUnicycleObstacleInputs,
        NumPyUnicycleObstacleCovariances,
    ]
):
    """Kalman Filter state estimator for unicycle model obstacles."""

    process_noise_covariance: ProcessNoiseCovarianceArray
    observation_noise_covariance: ObservationNoiseCovarianceArray
    model: NumPyUnicycleStateEstimationModel
    estimator: KalmanFilter

    @staticmethod
    def ekf(
        *,
        time_step_size: float,
        process_noise_covariance: NumPyNoiseCovarianceDescription[
            UnicycleEstimationD_x
        ],
        observation_noise_covariance: NumPyNoiseCovarianceDescription[
            UnicycleObservationD_o
        ],
        initial_state_covariance: EstimationStateCovarianceArray | None = None,
    ) -> "NumPyKfUnicycleStateEstimator":
        """Creates an EKF state estimator for the unicycle model with the specified noise
        covariances.

        Args:
            time_step_size: The time step size for the state transition model.
            process_noise_covariance: The process noise covariance, either as a full
                matrix, a vector of diagonal entries, or a scalar for isotropic noise.
            observation_noise_covariance: The observation noise covariance, either as a
                full matrix, a vector of diagonal entries, or a scalar for isotropic noise.
            initial_state_covariance: The initial state covariance for the Kalman filter.
                If not provided, low uncertainty will be assumed for observed states and high
                uncertainty for unobserved velocities.
        """
        return NumPyKfUnicycleStateEstimator(
            process_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                process_noise_covariance, dimension=UNICYCLE_ESTIMATION_D_X
            ),
            observation_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=UNICYCLE_OBSERVATION_D_O
            ),
            model=NumPyUnicycleStateEstimationModel.create(
                time_step_size=time_step_size,
                initial_state_covariance=initial_state_covariance,
            ),
            estimator=NumPyExtendedKalmanFilter.create(),
        )

    @staticmethod
    def ukf(
        *,
        time_step_size: float,
        process_noise_covariance: NumPyNoiseCovarianceDescription[
            UnicycleEstimationD_x
        ],
        observation_noise_covariance: NumPyNoiseCovarianceDescription[
            UnicycleObservationD_o
        ],
        initial_state_covariance: EstimationStateCovarianceArray | None = None,
    ) -> "NumPyKfUnicycleStateEstimator":
        """Creates a UKF state estimator for the unicycle model with the specified noise
        covariances.

        Args:
            time_step_size: The time step size for the state transition model.
            process_noise_covariance: The process noise covariance, either as a full
                matrix, a vector of diagonal entries, or a scalar for isotropic noise.
            observation_noise_covariance: The observation noise covariance, either as a
                full matrix, a vector of diagonal entries, or a scalar for isotropic noise.
            initial_state_covariance: The initial state covariance for the Kalman filter.
                If not provided, low uncertainty will be assumed for observed states and high
                uncertainty for unobserved velocities.
        """
        return NumPyKfUnicycleStateEstimator(
            process_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                process_noise_covariance, dimension=UNICYCLE_ESTIMATION_D_X
            ),
            observation_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=UNICYCLE_OBSERVATION_D_O
            ),
            model=NumPyUnicycleStateEstimationModel.create(
                time_step_size=time_step_size,
                initial_state_covariance=initial_state_covariance,
            ),
            estimator=NumPyUnscentedKalmanFilter.create(),
        )

    def estimate_from[K: int, T: int = int](
        self, history: NumPyUnicycleObstacleStatesHistory[T, K]
    ) -> EstimatedObstacleStates[
        NumPyUnicycleObstacleStates[K],
        NumPyUnicycleObstacleInputs[K],
        NumPyUnicycleObstacleCovariances[K],
    ]:
        estimate = cast(
            UnicycleGaussianBelief[K],
            self.estimator.filter(
                self.model.observations_from(history),
                initial_state_covariance=self.model.initial_state_covariance,
                state_transition=self.model,
                process_noise_covariance=self.process_noise_covariance,
                observation_noise_covariance=self.observation_noise_covariance,
                observation_matrix=self.model.observation_matrix,
            ),
        )

        return EstimatedObstacleStates(
            states=self.model.states_from(estimate),
            inputs=self.model.inputs_from(estimate),
            covariance=estimate.covariance,
        )


def simulate[T: int, N: int](
    inputs: ControlInputBatchArray[T, N],
    initial: StatesAtTimeStep[N],
    *,
    time_step_size: float,
    speed_limits: tuple[float, float],
    angular_velocity_limits: tuple[float, float],
) -> StateBatchArray[T, N]:
    horizon = inputs.shape[0]
    rollout_count = inputs.shape[2]
    states = np.zeros((horizon, UNICYCLE_D_X, rollout_count))
    current = initial

    for t in range(horizon):
        current = step(
            current,
            inputs[t],
            time_step_size=time_step_size,
            speed_limits=speed_limits,
            angular_velocity_limits=angular_velocity_limits,
        )
        states[t] = current

    assert shape_of(
        states, matches=(horizon, UNICYCLE_D_X, rollout_count), name="simulated states"
    )

    return states


def step[M: int](
    state: StatesAtTimeStep[M],
    control: ControlInputsAtTimeStep[M],
    *,
    time_step_size: float,
    speed_limits: tuple[float, float],
    angular_velocity_limits: tuple[float, float],
) -> StatesAtTimeStep[M]:
    x, y, theta = state[0], state[1], state[2]
    v, omega = control[0], control[1]
    linear_velocity = np.clip(v, *speed_limits)
    angular_velocity = np.clip(omega, *angular_velocity_limits)

    new_x = x + linear_velocity * np.cos(theta) * time_step_size
    new_y = y + linear_velocity * np.sin(theta) * time_step_size
    new_theta = theta + angular_velocity * time_step_size

    return np.stack([new_x, new_y, new_theta])
