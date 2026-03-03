from typing import Self, cast, Sequence, Final
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    jaxtyped,
    Array,
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
    DynamicalModel,
    ObstacleModel,
    ObstacleStateEstimator,
    EstimatedObstacleStates,
    NumPyGaussianBelief,
    NumPyNoiseModelProvider,
    NumPyNoiseCovarianceDescription,
)
from faran.filters import (
    NumPyExtendedKalmanFilter,
    NumPyUnscentedKalmanFilter,
    numpy_kalman_filter,
)
from faran.obstacles import NumPyObstacle2dPoses
from faran.models.common import SMALL_UNCERTAINTY, LARGE_UNCERTAINTY
from faran.models.basic import invalid_obstacle_filter_from

from numtypes import D, array

from jaxtyping import Float

import numpy as np


NO_LIMITS: Final = (float("-inf"), float("inf"))

BICYCLE_ESTIMATION_D_X: Final = 6
BICYCLE_OBSERVATION_D_O: Final = 3

type BicycleEstimationD_x = D[6]
type BicycleObservationD_o = D[3]

type StateArray = Float[Array, " BicycleD_x"]
type ControlInputSequenceArray = Float[Array, "T BicycleD_u"]
type StateBatchArray = Float[Array, "T BicycleD_x M"]
type ControlInputBatchArray = Float[Array, "T BicycleD_u M"]

type StatesAtTimeStep = Float[Array, "BicycleD_x M"]
type ControlInputsAtTimeStep = Float[Array, "BicycleD_u M"]

type EstimationStateCovarianceArray = Float[
    Array, "BicycleEstimationD_x BicycleEstimationD_x"
]
type ProcessNoiseCovarianceArray = Float[
    Array, "BicycleEstimationD_x BicycleEstimationD_x"
]
type ObservationNoiseCovarianceArray = Float[
    Array, "BicycleObservationD_o BicycleObservationD_o"
]
type ObservationMatrix = Float[Array, "BicycleObservationD_o BicycleEstimationD_x"]

type BicycleGaussianBelief = NumPyGaussianBelief
type NumPyBicycleObstacleCovariances = Float[
    Array, "BicycleEstimationD_x BicycleEstimationD_x K"
]
type KalmanFilter = NumPyExtendedKalmanFilter | NumPyUnscentedKalmanFilter


@jaxtyped
@dataclass(frozen=True)
class NumPyBicycleState(BicycleState, NumPyState):
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
        return cast(BicycleD_x, self.array.shape[0])

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
class NumPyBicycleStateSequence(BicycleStateSequence, NumPyStateSequence):
    batch: "NumPyBicycleStateBatch"
    rollout: int

    @staticmethod
    def of_states(states: Sequence[NumPyBicycleState]) -> "NumPyBicycleStateSequence":
        assert len(states) > 0, "States sequence must not be empty."

        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        return NumPyBicycleStateSequence(
            batch=NumPyBicycleStateBatch.wrap(array), rollout=0
        )

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T BicycleD_x"]:
        return self.array

    def step(self, index: int) -> NumPyBicycleState:
        return NumPyBicycleState(self.array[index])

    def batched(self) -> "NumPyBicycleStateBatch":
        return NumPyBicycleStateBatch.wrap(self.array[..., np.newaxis])

    def x(self) -> Float[Array, " T"]:
        return self.array[:, 0]

    def y(self) -> Float[Array, " T"]:
        return self.array[:, 1]

    def heading(self) -> Float[Array, " T"]:
        return self.array[:, 2]

    def speed(self) -> Float[Array, " T"]:
        return self.array[:, 3]

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_x:
        return cast(BicycleD_x, self.array.shape[1])

    @property
    def array(self) -> Float[Array, "T BicycleD_x"]:
        return self.batch.array[:, :, self.rollout]


@jaxtyped
@dataclass(frozen=True)
class NumPyBicycleStateBatch(BicycleStateBatch, NumPyStateBatch):
    _array: StateBatchArray

    @staticmethod
    def wrap(
        array: StateBatchArray,
    ) -> "NumPyBicycleStateBatch":
        return NumPyBicycleStateBatch(array)

    @staticmethod
    def of_states(states: Sequence[NumPyBicycleState]) -> "NumPyBicycleStateBatch":
        assert len(states) > 0, "States sequence must not be empty."

        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        return NumPyBicycleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> StateBatchArray:
        return self.array

    def heading(self) -> Float[Array, "T M"]:
        return self.array[:, 2, :]

    def speed(self) -> Float[Array, "T M"]:
        return self.array[:, 3, :]

    def rollout(self, index: int) -> NumPyBicycleStateSequence:
        return NumPyBicycleStateSequence(batch=self, rollout=index)

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_x:
        return cast(BicycleD_x, self.array.shape[1])

    @property
    def rollout_count(self) -> int:
        return self.array.shape[2]

    @property
    def positions(self) -> "NumPyBicyclePositions":
        return NumPyBicyclePositions(batch=self)

    @property
    def array(self) -> StateBatchArray:
        return self._array


@dataclass(frozen=True)
class NumPyBicyclePositions(BicyclePositions):
    batch: NumPyBicycleStateBatch

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T 2 M"]:
        return self.batch.array[:, :2, :]

    def x(self) -> Float[Array, "T M"]:
        return self.batch.array[:, 0, :]

    def y(self) -> Float[Array, "T M"]:
        return self.batch.array[:, 1, :]

    @property
    def horizon(self) -> int:
        return self.batch.horizon

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def rollout_count(self) -> int:
        return self.batch.rollout_count


@jaxtyped
@dataclass(frozen=True)
class NumPyBicycleControlInputSequence(
    BicycleControlInputSequence, NumPyControlInputSequence
):
    """Control inputs: [acceleration, steering_angle]."""

    _array: ControlInputSequenceArray

    @staticmethod
    def zeroes(horizon: int) -> "NumPyBicycleControlInputSequence":
        """Creates a zeroed control input sequence for the given horizon."""
        return NumPyBicycleControlInputSequence(np.zeros((horizon, BICYCLE_D_U)))

    def __array__(self, dtype: DataType | None = None) -> ControlInputSequenceArray:
        return self.array

    def similar(self, *, array: Float[Array, "L BicycleD_u"]) -> Self:
        return self.__class__(array)

    def accelerations(self) -> Float[Array, " T"]:
        return self.array[:, 0]

    def steering_angles(self) -> Float[Array, " T"]:
        return self.array[:, 1]

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_u:
        return cast(BicycleD_u, self.array.shape[1])

    @property
    def array(self) -> ControlInputSequenceArray:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyBicycleControlInputBatch(BicycleControlInputBatch, NumPyControlInputBatch):
    _array: ControlInputBatchArray

    @staticmethod
    def zero(
        *, horizon: int, rollout_count: int = 1
    ) -> "NumPyBicycleControlInputBatch":
        return NumPyBicycleControlInputBatch(
            np.zeros((horizon, BICYCLE_D_U, rollout_count))
        )

    @staticmethod
    def create(
        *, array: Float[Array, "T BicycleD_u M"]
    ) -> "NumPyBicycleControlInputBatch":
        return NumPyBicycleControlInputBatch(array)

    @staticmethod
    def of(
        sequence: NumPyBicycleControlInputSequence,
    ) -> "NumPyBicycleControlInputBatch":
        return NumPyBicycleControlInputBatch(sequence.array[..., np.newaxis])

    def __array__(self, dtype: DataType | None = None) -> ControlInputBatchArray:
        return self.array

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleD_u:
        return cast(BicycleD_u, self.array.shape[1])

    @property
    def rollout_count(self) -> int:
        return self.array.shape[2]

    @property
    def array(self) -> ControlInputBatchArray:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyBicycleObstacleStates:
    _array: Float[Array, "BicycleD_o K"]

    @staticmethod
    def wrap(
        array: Float[Array, "BicycleD_o K"],
    ) -> "NumPyBicycleObstacleStates":
        return NumPyBicycleObstacleStates(array)

    @staticmethod
    def create(
        *,
        x: Float[Array, " K"],
        y: Float[Array, " K"],
        heading: Float[Array, " K"],
        speed: Float[Array, " K"],
    ) -> "NumPyBicycleObstacleStates":
        return NumPyBicycleObstacleStates(np.stack([x, y, heading, speed], axis=0))

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "BicycleD_o K"]:
        return self.array

    def x(self) -> Float[Array, " K"]:
        return self.array[0, :]

    def y(self) -> Float[Array, " K"]:
        return self.array[1, :]

    def heading(self) -> Float[Array, " K"]:
        return self.array[2, :]

    def speed(self) -> Float[Array, " K"]:
        return self.array[3, :]

    @property
    def dimension(self) -> BicycleD_o:
        return cast(BicycleD_o, self.array.shape[0])

    @property
    def count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "BicycleD_o K"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyBicycleObstacleStateSequences:
    _array: Float[Array, "T BicycleEstimationD_x K"]
    _covariance: Float[Array, "T BicycleEstimationD_x BicycleEstimationD_x K"]

    @staticmethod
    def create(
        predictions: Sequence[BicycleGaussianBelief],
    ) -> "NumPyBicycleObstacleStateSequences":
        assert len(predictions) > 0, "Predictions sequence must not be empty."

        T_ = len(predictions)
        K_ = predictions[0].mean.shape[1]

        array = np.stack([belief.mean for belief in predictions], axis=0)
        covariance = np.stack([belief.covariance for belief in predictions], axis=0)

        assert array.shape == (T_, BICYCLE_ESTIMATION_D_X, K_)
        assert covariance.shape == (
            T_,
            BICYCLE_ESTIMATION_D_X,
            BICYCLE_ESTIMATION_D_X,
            K_,
        )

        return NumPyBicycleObstacleStateSequences(array, covariance)

    def __array__(
        self, dtype: DataType | None = None
    ) -> Float[Array, "T BicycleEstimationD_x K"]:
        return self.array

    def x(self) -> Float[Array, "T K"]:
        return self.array[:, 0, :]

    def y(self) -> Float[Array, "T K"]:
        return self.array[:, 1, :]

    def heading(self) -> Float[Array, "T K"]:
        return self.array[:, 2, :]

    def covariance(
        self,
    ) -> Float[Array, "T BicycleEstimationD_x BicycleEstimationD_x K"]:
        return self._covariance

    def pose_covariance(self) -> Float[Array, "T PoseD_o PoseD_o K"]:
        return self._covariance[:, :3, :3, :]

    def pose(self) -> NumPyObstacle2dPoses:
        return NumPyObstacle2dPoses.create(
            x=self.x(),
            y=self.y(),
            heading=self.heading(),
            covariance=self.pose_covariance(),
        )

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> BicycleEstimationD_x:
        return cast(BicycleEstimationD_x, self.array.shape[1])

    @property
    def count(self) -> int:
        return self.array.shape[2]

    @property
    def array(self) -> Float[Array, "T BicycleEstimationD_x K"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyBicycleObstacleInputs:
    _accelerations: Float[Array, " K"]
    _steering_angles: Float[Array, " K"]

    @staticmethod
    def wrap(inputs: Float[Array, "BicycleD_u K"]) -> "NumPyBicycleObstacleInputs":
        accelerations, steering_angles = inputs
        return NumPyBicycleObstacleInputs(
            _accelerations=accelerations, _steering_angles=steering_angles
        )

    @staticmethod
    def create(
        *, accelerations: Float[Array, " K"], steering_angles: Float[Array, " K"]
    ) -> "NumPyBicycleObstacleInputs":
        return NumPyBicycleObstacleInputs(
            _accelerations=accelerations,
            _steering_angles=steering_angles,
        )

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "BicycleD_u K"]:
        return self.array

    def accelerations(self) -> Float[Array, " K"]:
        return self._accelerations

    def steering_angles(self) -> Float[Array, " K"]:
        return self._steering_angles

    def zeroed(
        self, *, acceleration: bool = False, steering_angle: bool = False
    ) -> "NumPyBicycleObstacleInputs":
        """Returns a version of the inputs with the specified components zeroed out."""
        return NumPyBicycleObstacleInputs(
            _accelerations=np.zeros_like(self._accelerations)
            if acceleration
            else self._accelerations,
            _steering_angles=np.zeros_like(self._steering_angles)
            if steering_angle
            else self._steering_angles,
        )

    @property
    def dimension(self) -> BicycleD_u:
        return BICYCLE_D_U

    @property
    def count(self) -> int:
        return self._steering_angles.shape[0]

    @property
    def array(self) -> Float[Array, "BicycleD_u K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "BicycleD_u K"]:
        return np.stack([self._accelerations, self._steering_angles], axis=0)


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

    def simulate(
        self, inputs: NumPyBicycleControlInputBatch, initial_state: NumPyBicycleState
    ) -> NumPyBicycleStateBatch:
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

    def step(
        self, inputs: NumPyBicycleControlInputSequence, state: NumPyBicycleState
    ) -> NumPyBicycleState:
        state_as_rollouts = state.array.reshape(-1, 1)
        first_input = inputs.array[0].reshape(-1, 1)

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

    def forward(
        self, inputs: NumPyBicycleControlInputSequence, state: NumPyBicycleState
    ) -> NumPyBicycleStateSequence:
        return self.simulate(NumPyBicycleControlInputBatch.of(inputs), state).rollout(0)

    @property
    def time_step_size(self) -> float:
        return self._time_step_size


@dataclass(frozen=True)
class NumPyBicycleStateEstimationModel:
    """Kinematic bicycle model used for state estimation."""

    time_step_size: float
    wheelbase: float
    initial_state_covariance: EstimationStateCovarianceArray

    @staticmethod
    def create(
        *,
        time_step_size: float,
        wheelbase: float,
        initial_state_covariance: EstimationStateCovarianceArray | None = None,
    ) -> "NumPyBicycleStateEstimationModel":
        if initial_state_covariance is None:
            initial_state_covariance = (
                NumPyBicycleStateEstimationModel.default_initial_state_covariance()
            )

        return NumPyBicycleStateEstimationModel(
            time_step_size=time_step_size,
            wheelbase=wheelbase,
            initial_state_covariance=initial_state_covariance,
        )

    @staticmethod
    def default_initial_state_covariance() -> EstimationStateCovarianceArray:
        # NOTE: Sure of pose, unsure of everything else.
        # Steering angle covariance is kept somewhat smaller (0.1), since it
        # otherwise messes up UKF.
        return cast(
            EstimationStateCovarianceArray,
            np.diag(
                [
                    SMALL_UNCERTAINTY,
                    SMALL_UNCERTAINTY,
                    SMALL_UNCERTAINTY,
                    LARGE_UNCERTAINTY,
                    LARGE_UNCERTAINTY,
                    0.1,
                ]
            ),
        )

    def __call__(self, state: Float[Array, "D_x K"]) -> Float[Array, "D_x K"]:
        dt = self.time_step_size
        x, y, theta, v, a, delta = state

        return np.array(
            [
                x + v * np.cos(theta) * dt,
                y + v * np.sin(theta) * dt,
                theta + (v / self.wheelbase) * np.tan(delta) * dt,
                v + a * dt,
                a,
                delta,
            ]
        )

    def jacobian(self, state: Float[Array, "D_x K"]) -> Float[Array, "D_x D_x K"]:
        D_x, K = state.shape
        dt = self.time_step_size
        x, y, theta, v, a, delta = state

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        tan_delta = np.tan(delta)
        sec_delta_squared = 1 / np.cos(delta) ** 2

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
        jacobian[2, 3, :] = (1 / self.wheelbase) * tan_delta * dt  # ∂theta_next/∂v
        jacobian[2, 5, :] = (
            (v / self.wheelbase) * sec_delta_squared * dt
        )  # ∂theta_next/∂delta

        # Partial derivatives for v
        jacobian[3, 3, :] = 1  # ∂v_next/∂v
        jacobian[3, 4, :] = dt  # ∂v_next/∂a

        # Partial derivatives for a and delta are identity since they are assumed constant
        jacobian[4, 4, :] = 1  # ∂a_next/∂a
        jacobian[5, 5, :] = 1  # ∂delta_next/∂delta

        return jacobian

    def observations_from(
        self, history: NumPyBicycleObstacleStatesHistory
    ) -> Float[Array, "T BicycleObservationD_o K"]:
        return np.stack([history.x(), history.y(), history.heading()], axis=1)

    def states_from(self, belief: BicycleGaussianBelief) -> NumPyBicycleObstacleStates:
        return NumPyBicycleObstacleStates.wrap(belief.mean[:4, :])

    def inputs_from(self, belief: BicycleGaussianBelief) -> NumPyBicycleObstacleInputs:
        return NumPyBicycleObstacleInputs.wrap(belief.mean[4:, :])

    def initial_belief_from(
        self,
        *,
        states: NumPyBicycleObstacleStates,
        inputs: NumPyBicycleObstacleInputs,
        covariances: NumPyBicycleObstacleCovariances | None = None,
    ) -> BicycleGaussianBelief:
        augmented = np.concatenate([states.array, inputs.array], axis=0)

        if covariances is None:
            # NOTE: No covariance means we are "certain" about the states.
            covariances = np.broadcast_to(  # type: ignore
                np.eye(BICYCLE_ESTIMATION_D_X)[:, :, np.newaxis] * SMALL_UNCERTAINTY,
                (BICYCLE_ESTIMATION_D_X, BICYCLE_ESTIMATION_D_X, states.count),
            ).copy()

        return NumPyGaussianBelief(mean=augmented, covariance=covariances)

    @cached_property
    def observation_matrix(self) -> ObservationMatrix:
        # NOTE: We observe the pose directly.
        return np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )


@dataclass(kw_only=True, frozen=True)
class NumPyBicycleObstacleModel(
    ObstacleModel[
        NumPyBicycleObstacleStatesHistory,
        NumPyBicycleObstacleStates,
        NumPyBicycleObstacleInputs,
        NumPyBicycleObstacleCovariances,
        NumPyBicycleObstacleStateSequences,
    ]
):
    """Propagates bicycle kinematics forward given states and control inputs."""

    model: NumPyBicycleStateEstimationModel
    process_noise_covariance: ProcessNoiseCovarianceArray
    predictor: KalmanFilter

    @staticmethod
    def unscented(
        *,
        time_step_size: float,
        wheelbase: float = 1.0,
        process_noise_covariance: NumPyNoiseCovarianceDescription = 1e-3,
        sigma_point_spread: float = 1.0,
        prior_knowledge: float = 2.0,
    ) -> "NumPyBicycleObstacleModel":
        """Creates a bicycle obstacle model for obstacle state prediction. This
        model uses the unscented transform for propagating state information through
        the nonlinear bicycle dynamics.

        Args:
            time_step_size: Time step size for state propagation.
            wheelbase: The wheelbase length of the bicycle.
            process_noise_covariance: The process noise covariance, either as a
                full covariance array, a diagonal covariance vector, or a scalar
                variance representing isotropic noise.
            sigma_point_spread: Spread of sigma points (α) for the unscented
                transform, controlling how far the sigma points are from the mean.
            prior_knowledge: Prior knowledge (β) about the state distribution
                for the unscented transform.
        """
        return NumPyBicycleObstacleModel(
            model=NumPyBicycleStateEstimationModel.create(
                time_step_size=time_step_size, wheelbase=wheelbase
            ),
            process_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                process_noise_covariance, dimension=BICYCLE_ESTIMATION_D_X
            ),
            predictor=NumPyUnscentedKalmanFilter.create(
                alpha=sigma_point_spread, beta=prior_knowledge
            ),
        )

    def forward(
        self,
        *,
        states: NumPyBicycleObstacleStates,
        inputs: NumPyBicycleObstacleInputs,
        covariances: NumPyBicycleObstacleCovariances | None,
        horizon: int,
    ) -> NumPyBicycleObstacleStateSequences:
        beliefs = []
        last = self.model.initial_belief_from(
            states=states, inputs=inputs, covariances=covariances
        )

        for _ in range(horizon):
            beliefs.append(
                last := self.predictor.predict(
                    belief=last,
                    state_transition=self.model,
                    process_noise_covariance=self.process_noise_covariance,
                )
            )

        return NumPyBicycleObstacleStateSequences.create(beliefs)


@dataclass(frozen=True)
class NumPyFiniteDifferenceBicycleStateEstimator(
    ObstacleStateEstimator[
        NumPyBicycleObstacleStatesHistory,
        NumPyBicycleObstacleStates,
        NumPyBicycleObstacleInputs,
    ]
):
    time_step_size: float
    wheelbase: float

    @staticmethod
    def create(
        *, time_step_size: float, wheelbase: float
    ) -> "NumPyFiniteDifferenceBicycleStateEstimator":
        return NumPyFiniteDifferenceBicycleStateEstimator(
            time_step_size=time_step_size, wheelbase=wheelbase
        )

    def estimate_from(
        self, history: NumPyBicycleObstacleStatesHistory
    ) -> EstimatedObstacleStates[
        NumPyBicycleObstacleStates, NumPyBicycleObstacleInputs, None
    ]:
        """Estimates current states and inputs from pose history using finite differences. Zeros
        are assumed for states that cannot be estimated due to insufficient history length.

        Computes the following quantities from the kinematic bicycle model:

        **Speed** (requires T ≥ 2):
            Projection of displacement onto the heading direction (negative for reverse):
            $$v_t = \\frac{(x_t - x_{t-1}) \\cos(\\theta_t) + (y_t - y_{t-1}) \\sin(\\theta_t)}{\\Delta t}$$

        **Acceleration** (requires T ≥ 3):
            Change in speed over time:
            $$a_t = \\frac{v_t - v_{t-1}}{\\Delta t}$$

        **Steering angle** (requires T ≥ 2):
            Derived from the kinematic bicycle model, where $\\omega = \\frac{d\\theta}{dt}$ is the heading rate:
            $$\\delta_t = \\arctan\\left(\\frac{L \\cdot \\omega_t}{v_t}\\right)$$
            Zero when $|v_t| < \\epsilon$ (estimate becomes unreliable).

        Args:
            history: Obstacle pose history containing at least one entry.
        """
        assert history.horizon > 0, "History must contain at least one state."

        filter_invalid_short = invalid_obstacle_filter_from(history, check_recent=2)
        filter_invalid_long = invalid_obstacle_filter_from(history, check_recent=3)

        speeds = self.estimate_speeds_from(history)
        accelerations = self.estimate_accelerations_from(history, speeds=speeds)
        steering_angles = self.estimate_steering_angles_from(history, speeds=speeds)

        return EstimatedObstacleStates(
            states=NumPyBicycleObstacleStates.create(
                x=history.x()[-1],
                y=history.y()[-1],
                heading=history.heading()[-1],
                speed=filter_invalid_short(speeds),
            ),
            inputs=NumPyBicycleObstacleInputs.create(
                accelerations=filter_invalid_long(accelerations),
                steering_angles=filter_invalid_short(steering_angles),
            ),
            covariance=None,
        )

    def invalid_obstacle_mask_from(
        self, history: NumPyBicycleObstacleStatesHistory
    ) -> Float[Array, " K"]:
        return np.any(
            np.isnan(history.x()[-3:])
            | np.isnan(history.y()[-3:])
            | np.isnan(history.heading()[-3:]),
            axis=0,
        )

    def estimate_speeds_from(
        self, history: NumPyBicycleObstacleStatesHistory
    ) -> Float[Array, " K"]:
        if history.horizon < 2:
            return np.zeros((history.count,))

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

    def estimate_accelerations_from(
        self,
        history: NumPyBicycleObstacleStatesHistory,
        *,
        speeds: Float[Array, " K"],
    ) -> Float[Array, " K"]:
        if history.horizon < 3:
            return np.zeros((history.count,))

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

    def estimate_steering_angles_from(
        self, history: NumPyBicycleObstacleStatesHistory, *, speeds: Float[Array, " K"]
    ) -> Float[Array, " K"]:
        if history.horizon < 2:
            return np.zeros((history.count,))

        heading = history.heading()

        return self._estimate_steering_angles_from(
            heading_current=heading[-1], heading_previous=heading[-2], speeds=speeds
        )

    def _estimate_speeds_from(
        self,
        *,
        x_current: Float[Array, " K"],
        y_current: Float[Array, " K"],
        x_previous: Float[Array, " K"],
        y_previous: Float[Array, " K"],
        heading_current: Float[Array, " K"],
    ) -> Float[Array, " K"]:
        delta_x = x_current - x_previous
        delta_y = y_current - y_previous

        speeds = (
            delta_x * np.cos(heading_current) + delta_y * np.sin(heading_current)
        ) / self.time_step_size

        assert speeds.shape == (x_current.shape[0],)

        return speeds

    def _estimate_accelerations_from(
        self, *, speeds_current: Float[Array, " K"], speeds_previous: Float[Array, " K"]
    ) -> Float[Array, " K"]:
        return (speeds_current - speeds_previous) / self.time_step_size

    def _estimate_steering_angles_from(
        self,
        *,
        heading_current: Float[Array, " K"],
        heading_previous: Float[Array, " K"],
        speeds: Float[Array, " K"],
    ) -> Float[Array, " K"]:
        heading_velocity = (heading_current - heading_previous) / self.time_step_size

        with np.errstate(invalid="ignore"):
            steering_angles = np.where(
                np.abs(speeds) > 1e-6,
                np.arctan(heading_velocity * self.wheelbase / speeds),
                0.0,
            )

        return steering_angles


@dataclass(frozen=True)
class NumPyKfBicycleStateEstimator(
    ObstacleStateEstimator[
        NumPyBicycleObstacleStatesHistory,
        NumPyBicycleObstacleStates,
        NumPyBicycleObstacleInputs,
        NumPyBicycleObstacleCovariances,
    ]
):
    """Kalman Filter state estimator for bicycle model obstacles."""

    process_noise_covariance: ProcessNoiseCovarianceArray
    observation_noise_covariance: ObservationNoiseCovarianceArray
    model: NumPyBicycleStateEstimationModel
    estimator: KalmanFilter
    noise_model: NumPyNoiseModelProvider | None = None

    @staticmethod
    def ekf(
        *,
        time_step_size: float,
        wheelbase: float,
        process_noise_covariance: NumPyNoiseCovarianceDescription,
        observation_noise_covariance: NumPyNoiseCovarianceDescription,
        initial_state_covariance: EstimationStateCovarianceArray | None = None,
        noise_model: NumPyNoiseModelProvider | None = None,
    ) -> "NumPyKfBicycleStateEstimator":
        """Creates an EKF state estimator for the bicycle model with the specified noise
        covariances.

        Args:
            time_step_size: The time step size for the state transition model.
            wheelbase: The wheelbase of the bicycle.
            process_noise_covariance: The process noise covariance, either as a full
                matrix, a vector of diagonal entries, or a scalar for isotropic noise.
            observation_noise_covariance: The observation noise covariance, either as a
                full matrix, a vector of diagonal entries, or a scalar for isotropic noise.
            initial_state_covariance: The initial state covariance for the Kalman filter.
                If not provided, low uncertainty will be assumed for observed states and high
                uncertainty for speed and inputs.
            noise_model: Optional noise model provider for adaptive noise estimation.
        """
        return NumPyKfBicycleStateEstimator(
            process_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                process_noise_covariance, dimension=BICYCLE_ESTIMATION_D_X
            ),
            observation_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=BICYCLE_OBSERVATION_D_O
            ),
            model=NumPyBicycleStateEstimationModel.create(
                time_step_size=time_step_size,
                wheelbase=wheelbase,
                initial_state_covariance=initial_state_covariance,
            ),
            estimator=NumPyExtendedKalmanFilter.create(noise_model=noise_model),
        )

    @staticmethod
    def ukf(
        *,
        time_step_size: float,
        wheelbase: float,
        process_noise_covariance: NumPyNoiseCovarianceDescription,
        observation_noise_covariance: NumPyNoiseCovarianceDescription,
        initial_state_covariance: EstimationStateCovarianceArray | None = None,
        noise_model: NumPyNoiseModelProvider | None = None,
    ) -> "NumPyKfBicycleStateEstimator":
        """Creates a UKF state estimator for the bicycle model with the specified noise
        covariances.

        Args:
            time_step_size: The time step size for the state transition model.
            wheelbase: The wheelbase of the bicycle.
            process_noise_covariance: The process noise covariance, either as a full
                matrix, a vector of diagonal entries, or a scalar for isotropic noise.
            observation_noise_covariance: The observation noise covariance, either as a
                full matrix, a vector of diagonal entries, or a scalar for isotropic noise.
            initial_state_covariance: The initial state covariance for the Kalman filter.
                If not provided, low uncertainty will be assumed for observed states and high
                uncertainty for speed and inputs.
            noise_model: Optional noise model provider for adaptive noise estimation.
        """
        return NumPyKfBicycleStateEstimator(
            process_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                process_noise_covariance, dimension=BICYCLE_ESTIMATION_D_X
            ),
            observation_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=BICYCLE_OBSERVATION_D_O
            ),
            model=NumPyBicycleStateEstimationModel.create(
                time_step_size=time_step_size,
                wheelbase=wheelbase,
                initial_state_covariance=initial_state_covariance,
            ),
            estimator=NumPyUnscentedKalmanFilter.create(noise_model=noise_model),
        )

    def estimate_from(
        self, history: NumPyBicycleObstacleStatesHistory
    ) -> EstimatedObstacleStates[
        NumPyBicycleObstacleStates,
        NumPyBicycleObstacleInputs,
        NumPyBicycleObstacleCovariances,
    ]:
        estimate = cast(
            BicycleGaussianBelief,
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


def simulate(
    inputs: ControlInputBatchArray,
    initial: StatesAtTimeStep,
    *,
    time_step_size: float,
    wheelbase: float,
    speed_limits: tuple[float, float],
    steering_limits: tuple[float, float],
    acceleration_limits: tuple[float, float],
) -> StateBatchArray:
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

    return states


def step(
    state: StatesAtTimeStep,
    control: ControlInputsAtTimeStep,
    *,
    time_step_size: float,
    wheelbase: float,
    speed_limits: tuple[float, float],
    steering_limits: tuple[float, float],
    acceleration_limits: tuple[float, float],
) -> StatesAtTimeStep:
    x, y, theta, v = state[0], state[1], state[2], state[3]
    a, delta = control[0], control[1]
    acceleration = np.clip(a, *acceleration_limits)
    steering = np.clip(delta, *steering_limits)

    new_x = x + v * np.cos(theta) * time_step_size
    new_y = y + v * np.sin(theta) * time_step_size
    new_theta = theta + v * np.tan(steering) / wheelbase * time_step_size
    new_v = np.clip(v + acceleration * time_step_size, *speed_limits)

    return np.stack([new_x, new_y, new_theta, new_v])
