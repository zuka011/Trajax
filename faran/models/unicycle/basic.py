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
    DynamicalModel,
    ObstacleModel,
    ObstacleStateEstimator,
    EstimatedObstacleStates,
    NumPyGaussianBelief,
    NumPyNoiseCovarianceDescription,
    NumPyNoiseModelProvider,
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

UNICYCLE_ESTIMATION_D_X: Final = 5
UNICYCLE_OBSERVATION_D_O: Final = 3

type UnicycleEstimationD_x = D[5]
type UnicycleObservationD_o = D[3]

type StateArray = Float[Array, " UnicycleD_x"]
type ControlInputSequenceArray = Float[Array, "T UnicycleD_u"]
type StateBatchArray = Float[Array, "T UnicycleD_x M"]
type ControlInputBatchArray = Float[Array, "T UnicycleD_u M"]

type StatesAtTimeStep = Float[Array, "UnicycleD_x M"]
type ControlInputsAtTimeStep = Float[Array, "UnicycleD_u M"]

type EstimationStateCovarianceArray = Float[
    Array, "UnicycleEstimationD_x UnicycleEstimationD_x"
]
type ProcessNoiseCovarianceArray = Float[
    Array, "UnicycleEstimationD_x UnicycleEstimationD_x"
]
type ObservationNoiseCovarianceArray = Float[
    Array, "UnicycleObservationD_o UnicycleObservationD_o"
]
type ObservationMatrix = Float[Array, "UnicycleObservationD_o UnicycleEstimationD_x"]

type UnicycleGaussianBelief = NumPyGaussianBelief
type NumPyUnicycleObstacleCovariances = Float[
    Array, "UnicycleEstimationD_x UnicycleEstimationD_x K"
]
type KalmanFilter = NumPyExtendedKalmanFilter | NumPyUnscentedKalmanFilter


@jaxtyped
@dataclass(frozen=True)
class NumPyUnicycleState(UnicycleState, NumPyState):
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
        return cast(UnicycleD_x, self.array.shape[0])

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
class NumPyUnicycleStateSequence(UnicycleStateSequence, NumPyStateSequence):
    batch: "NumPyUnicycleStateBatch"
    rollout: int

    @staticmethod
    def of_states(states: Sequence[NumPyUnicycleState]) -> "NumPyUnicycleStateSequence":
        assert len(states) > 0, "States sequence must not be empty."

        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        return NumPyUnicycleStateSequence(
            batch=NumPyUnicycleStateBatch.wrap(array), rollout=0
        )

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T UnicycleD_x"]:
        return self.array

    def step(self, index: int) -> NumPyUnicycleState:
        return NumPyUnicycleState(self.array[index])

    def batched(self) -> "NumPyUnicycleStateBatch":
        return NumPyUnicycleStateBatch.wrap(self.array[..., np.newaxis])

    def x(self) -> Float[Array, " T"]:
        return self.array[:, 0]

    def y(self) -> Float[Array, " T"]:
        return self.array[:, 1]

    def heading(self) -> Float[Array, " T"]:
        return self.array[:, 2]

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> UnicycleD_x:
        return cast(UnicycleD_x, self.array.shape[1])

    @property
    def array(self) -> Float[Array, "T UnicycleD_x"]:
        return self.batch.array[:, :, self.rollout]


@jaxtyped
@dataclass(frozen=True)
class NumPyUnicycleStateBatch(UnicycleStateBatch, NumPyStateBatch):
    _array: StateBatchArray

    @staticmethod
    def wrap(
        array: StateBatchArray,
    ) -> "NumPyUnicycleStateBatch":
        return NumPyUnicycleStateBatch(array)

    @staticmethod
    def of_states(states: Sequence[NumPyUnicycleState]) -> "NumPyUnicycleStateBatch":
        assert len(states) > 0, "States sequence must not be empty."

        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        return NumPyUnicycleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> StateBatchArray:
        return self.array

    def heading(self) -> Float[Array, "T M"]:
        return self.array[:, 2, :]

    def rollout(self, index: int) -> NumPyUnicycleStateSequence:
        return NumPyUnicycleStateSequence(batch=self, rollout=index)

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> UnicycleD_x:
        return cast(UnicycleD_x, self.array.shape[1])

    @property
    def rollout_count(self) -> int:
        return self.array.shape[2]

    @property
    def positions(self) -> "NumPyUnicyclePositions":
        return NumPyUnicyclePositions(batch=self)

    @property
    def array(self) -> StateBatchArray:
        return self._array


@dataclass(frozen=True)
class NumPyUnicyclePositions(UnicyclePositions):
    batch: NumPyUnicycleStateBatch

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
class NumPyUnicycleControlInputSequence(
    UnicycleControlInputSequence, NumPyControlInputSequence
):
    """Control inputs: [linear velocity, angular velocity]."""

    _array: ControlInputSequenceArray

    @staticmethod
    def zeroes(horizon: int) -> "NumPyUnicycleControlInputSequence":
        """Creates a zeroed control input sequence for the given horizon."""
        return NumPyUnicycleControlInputSequence(np.zeros((horizon, UNICYCLE_D_U)))

    def __array__(self, dtype: DataType | None = None) -> ControlInputSequenceArray:
        return self.array

    def similar(
        self, *, array: Float[Array, "L UnicycleD_u"]
    ) -> "Self | NumPyUnicycleControlInputSequence":
        return self.__class__(array)

    def linear_velocities(self) -> Float[Array, " T"]:
        return self.array[:, 0]

    def angular_velocities(self) -> Float[Array, " T"]:
        return self.array[:, 1]

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> UnicycleD_u:
        return cast(UnicycleD_u, self.array.shape[1])

    @property
    def array(self) -> ControlInputSequenceArray:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyUnicycleControlInputBatch(UnicycleControlInputBatch, NumPyControlInputBatch):
    _array: ControlInputBatchArray

    @staticmethod
    def zero(
        *, horizon: int, rollout_count: int = 1
    ) -> "NumPyUnicycleControlInputBatch":
        return NumPyUnicycleControlInputBatch(
            np.zeros((horizon, UNICYCLE_D_U, rollout_count))
        )

    @staticmethod
    def create(
        *, array: Float[Array, "T UnicycleD_u M"]
    ) -> "NumPyUnicycleControlInputBatch":
        return NumPyUnicycleControlInputBatch(array)

    @staticmethod
    def of(
        sequence: NumPyUnicycleControlInputSequence,
    ) -> "NumPyUnicycleControlInputBatch":
        return NumPyUnicycleControlInputBatch(sequence.array[..., np.newaxis])

    def __array__(self, dtype: DataType | None = None) -> ControlInputBatchArray:
        return self.array

    def linear_velocity(self) -> Float[Array, "T M"]:
        return self.array[:, 0, :]

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> UnicycleD_u:
        return cast(UnicycleD_u, self.array.shape[1])

    @property
    def rollout_count(self) -> int:
        return self.array.shape[2]

    @property
    def array(self) -> ControlInputBatchArray:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyUnicycleObstacleStates:
    _array: Float[Array, "UnicycleD_o K"]

    @staticmethod
    def wrap(array: Float[Array, "UnicycleD_o K"]) -> "NumPyUnicycleObstacleStates":
        return NumPyUnicycleObstacleStates(array)

    @staticmethod
    def create(
        *, x: Float[Array, " K"], y: Float[Array, " K"], heading: Float[Array, " K"]
    ) -> "NumPyUnicycleObstacleStates":
        return NumPyUnicycleObstacleStates(np.stack([x, y, heading], axis=0))

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "UnicycleD_o K"]:
        return self.array

    def x(self) -> Float[Array, " K"]:
        return self.array[0, :]

    def y(self) -> Float[Array, " K"]:
        return self.array[1, :]

    def heading(self) -> Float[Array, " K"]:
        return self.array[2, :]

    @property
    def dimension(self) -> UnicycleD_o:
        return cast(UnicycleD_o, self.array.shape[0])

    @property
    def count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "UnicycleD_o K"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyUnicycleObstacleStateSequences:
    _array: Float[Array, "T UnicycleEstimationD_x K"]
    _covariance: Float[Array, "T UnicycleEstimationD_x UnicycleEstimationD_x K"]

    @staticmethod
    def create(
        predictions: Sequence[UnicycleGaussianBelief],
    ) -> "NumPyUnicycleObstacleStateSequences":
        assert len(predictions) > 0, "Predictions sequence must not be empty."

        return NumPyUnicycleObstacleStateSequences(
            _array=np.stack([belief.mean for belief in predictions], axis=0),
            _covariance=np.stack([belief.covariance for belief in predictions], axis=0),
        )

    def __array__(
        self, dtype: DataType | None = None
    ) -> Float[Array, "T UnicycleEstimationD_x K"]:
        return self.array

    def x(self) -> Float[Array, "T K"]:
        return self.array[:, 0, :]

    def y(self) -> Float[Array, "T K"]:
        return self.array[:, 1, :]

    def heading(self) -> Float[Array, "T K"]:
        return self.array[:, 2, :]

    def covariance(
        self,
    ) -> Float[Array, "T UnicycleEstimationD_x UnicycleEstimationD_x K"]:
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
    def dimension(self) -> UnicycleEstimationD_x:
        return cast(UnicycleEstimationD_x, self.array.shape[1])

    @property
    def count(self) -> int:
        return self.array.shape[2]

    @property
    def array(self) -> Float[Array, "T UnicycleEstimationD_x K"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPyUnicycleObstacleInputs:
    _linear_velocities: Float[Array, " K"]
    _angular_velocities: Float[Array, " K"]

    @staticmethod
    def wrap(
        array: Float[Array, "UnicycleD_u K"],
    ) -> "NumPyUnicycleObstacleInputs":
        linear, angular = array
        return NumPyUnicycleObstacleInputs(
            _linear_velocities=linear, _angular_velocities=angular
        )

    @staticmethod
    def create(
        *, linear_velocities: Float[Array, " K"], angular_velocities: Float[Array, " K"]
    ) -> "NumPyUnicycleObstacleInputs":
        return NumPyUnicycleObstacleInputs(
            _linear_velocities=linear_velocities, _angular_velocities=angular_velocities
        )

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "UnicycleD_u K"]:
        return self._array

    def linear_velocities(self) -> Float[Array, " K"]:
        return self._linear_velocities

    def angular_velocities(self) -> Float[Array, " K"]:
        return self._angular_velocities

    def zeroed(
        self, *, linear_velocity: bool = False, angular_velocity: bool = False
    ) -> "NumPyUnicycleObstacleInputs":
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
    def count(self) -> int:
        return self._linear_velocities.shape[0]

    @property
    def array(self) -> Float[Array, "UnicycleD_u K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[Array, "UnicycleD_u K"]:
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

    def simulate(
        self, inputs: NumPyUnicycleControlInputBatch, initial_state: NumPyUnicycleState
    ) -> NumPyUnicycleStateBatch:
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

    def step(
        self, inputs: NumPyUnicycleControlInputSequence, state: NumPyUnicycleState
    ) -> NumPyUnicycleState:
        state_as_rollouts = state.array.reshape(-1, 1)
        first_input = inputs.array[0].reshape(-1, 1)

        return NumPyUnicycleState(
            step(
                state_as_rollouts,
                first_input,
                time_step_size=self.time_step_size,
                speed_limits=self.speed_limits,
                angular_velocity_limits=self.angular_velocity_limits,
            )[:, 0]
        )

    def forward(
        self, inputs: NumPyUnicycleControlInputSequence, state: NumPyUnicycleState
    ) -> NumPyUnicycleStateSequence:
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

    def __call__(self, state: Float[Array, "D_x K"]) -> Float[Array, "D_x K"]:
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

    def jacobian(self, state: Float[Array, "D_x K"]) -> Float[Array, "D_x D_x K"]:
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

        return jacobian

    def observations_from(
        self, history: NumPyUnicycleObstacleStatesHistory
    ) -> Float[Array, "T UnicycleObservationD_o K"]:
        return np.stack([history.x(), history.y(), history.heading()], axis=1)

    def states_from(
        self, belief: UnicycleGaussianBelief
    ) -> NumPyUnicycleObstacleStates:
        return NumPyUnicycleObstacleStates.wrap(belief.mean[:3, :])

    def inputs_from(
        self, belief: UnicycleGaussianBelief
    ) -> NumPyUnicycleObstacleInputs:
        return NumPyUnicycleObstacleInputs.wrap(belief.mean[3:, :])

    def initial_belief_from(
        self,
        *,
        states: NumPyUnicycleObstacleStates,
        inputs: NumPyUnicycleObstacleInputs,
        covariances: NumPyUnicycleObstacleCovariances | None = None,
    ) -> UnicycleGaussianBelief:
        augmented = np.concatenate([states.array, inputs.array], axis=0)

        if covariances is None:
            # NOTE: No covariance means we are "certain" about the states.
            covariances = np.broadcast_to(  # type: ignore
                np.eye(UNICYCLE_ESTIMATION_D_X)[:, :, np.newaxis] * SMALL_UNCERTAINTY,
                (UNICYCLE_ESTIMATION_D_X, UNICYCLE_ESTIMATION_D_X, states.count),
            ).copy()

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

    def forward(
        self,
        *,
        states: NumPyUnicycleObstacleStates,
        inputs: NumPyUnicycleObstacleInputs,
        covariances: NumPyUnicycleObstacleCovariances | None,
        horizon: int,
    ) -> NumPyUnicycleObstacleStateSequences:
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

    def estimate_from(
        self, history: NumPyUnicycleObstacleStatesHistory
    ) -> EstimatedObstacleStates[
        NumPyUnicycleObstacleStates, NumPyUnicycleObstacleInputs, None
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

    def estimate_speeds_from(
        self, history: NumPyUnicycleObstacleStatesHistory
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

    def estimate_angular_velocities_from(
        self, history: NumPyUnicycleObstacleStatesHistory
    ) -> Float[Array, " K"]:
        if history.horizon < 2:
            return np.zeros((history.count,))

        heading = history.heading()

        return self._estimate_angular_velocities_from(
            heading_current=heading[-1],
            heading_previous=heading[-2],
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

        return speeds

    def _estimate_angular_velocities_from(
        self,
        *,
        heading_current: Float[Array, " K"],
        heading_previous: Float[Array, " K"],
    ) -> Float[Array, " K"]:
        angular_velocities = (heading_current - heading_previous) / self.time_step_size

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
        process_noise_covariance: NumPyNoiseCovarianceDescription,
        observation_noise_covariance: NumPyNoiseCovarianceDescription,
        initial_state_covariance: EstimationStateCovarianceArray | None = None,
        noise_model: NumPyNoiseModelProvider | None = None,
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
            estimator=NumPyExtendedKalmanFilter.create(noise_model=noise_model),
        )

    @staticmethod
    def ukf(
        *,
        time_step_size: float,
        process_noise_covariance: NumPyNoiseCovarianceDescription,
        observation_noise_covariance: NumPyNoiseCovarianceDescription,
        initial_state_covariance: EstimationStateCovarianceArray | None = None,
        noise_model: NumPyNoiseModelProvider | None = None,
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
            estimator=NumPyUnscentedKalmanFilter.create(noise_model=noise_model),
        )

    def estimate_from(
        self, history: NumPyUnicycleObstacleStatesHistory
    ) -> EstimatedObstacleStates[
        NumPyUnicycleObstacleStates,
        NumPyUnicycleObstacleInputs,
        NumPyUnicycleObstacleCovariances,
    ]:
        estimate = self.estimator.filter(
            self.model.observations_from(history),
            initial_state_covariance=self.model.initial_state_covariance,
            state_transition=self.model,
            process_noise_covariance=self.process_noise_covariance,
            observation_noise_covariance=self.observation_noise_covariance,
            observation_matrix=self.model.observation_matrix,
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
    speed_limits: tuple[float, float],
    angular_velocity_limits: tuple[float, float],
) -> StateBatchArray:
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

    return states


def step(
    state: StatesAtTimeStep,
    control: ControlInputsAtTimeStep,
    *,
    time_step_size: float,
    speed_limits: tuple[float, float],
    angular_velocity_limits: tuple[float, float],
) -> StatesAtTimeStep:
    x, y, theta = state[0], state[1], state[2]
    v, omega = control[0], control[1]
    linear_velocity = np.clip(v, *speed_limits)
    angular_velocity = np.clip(omega, *angular_velocity_limits)

    new_x = x + linear_velocity * np.cos(theta) * time_step_size
    new_y = y + linear_velocity * np.sin(theta) * time_step_size
    new_theta = theta + angular_velocity * time_step_size

    return np.stack([new_x, new_y, new_theta])
