from typing import cast, overload, Self, Sequence, Final, Any, NamedTuple
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    jaxtyped,
    DataType,
    JaxState,
    JaxStateSequence,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxUnicycleObstacleStatesHistory,
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
    POSE_D_O,
    DynamicalModel,
    ObstacleModel,
    ObstacleStateEstimator,
    EstimatedObstacleStates,
)
from faran.filters import (
    JaxExtendedKalmanFilter,
    JaxGaussianBelief,
    JaxNoiseCovarianceDescription,
    JaxUnscentedKalmanFilter,
    jax_kalman_filter,
)
from faran.obstacles import JaxObstacle2dPoses
from faran.models.common import SMALL_UNCERTAINTY, LARGE_UNCERTAINTY
from faran.models.accelerated import invalid_obstacle_filter_from

from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims, D

import jax
import jax.numpy as jnp
import numpy as np

NO_LIMITS: Final = (jnp.asarray(-jnp.inf), jnp.asarray(jnp.inf))

UNICYCLE_ESTIMATION_D_X: Final = 5
UNICYCLE_OBSERVATION_D_O: Final = 3

type UnicycleEstimationD_x = D[5]
type UnicycleObservationD_o = D[3]

type StateArray = Float[JaxArray, f"{UNICYCLE_D_X}"]
type ControlInputSequenceArray[T: int] = Float[JaxArray, f"T {UNICYCLE_D_U}"]
type StateBatchArray[T: int, M: int] = Float[JaxArray, f"T {UNICYCLE_D_X} M"]
type ControlInputBatchArray[T: int, M: int] = Float[JaxArray, f"T {UNICYCLE_D_U} M"]

type StatesAtTimeStep[M: int] = Float[JaxArray, f"{UNICYCLE_D_X} M"]
type ControlInputsAtTimeStep[M: int] = Float[JaxArray, f"{UNICYCLE_D_U} M"]

type EstimationStateCovarianceArray = Float[
    JaxArray, f"{UNICYCLE_ESTIMATION_D_X} {UNICYCLE_ESTIMATION_D_X}"
]
type ProcessNoiseCovarianceArray = Float[
    JaxArray, f"{UNICYCLE_ESTIMATION_D_X} {UNICYCLE_ESTIMATION_D_X}"
]
type ObservationNoiseCovarianceArray = Float[
    JaxArray, f"{UNICYCLE_OBSERVATION_D_O} {UNICYCLE_OBSERVATION_D_O}"
]
type ObservationMatrix = Float[
    JaxArray, f"{UNICYCLE_OBSERVATION_D_O} {UNICYCLE_ESTIMATION_D_X}"
]

type UnicycleGaussianBelief[K: int] = JaxGaussianBelief
type JaxUnicycleObstacleCovariances[K: int] = Float[
    JaxArray, f"{UNICYCLE_ESTIMATION_D_X} {UNICYCLE_ESTIMATION_D_X} K"
]
type KalmanFilter = JaxExtendedKalmanFilter | JaxUnscentedKalmanFilter


@jaxtyped
@dataclass(frozen=True)
class JaxUnicycleState(UnicycleState, JaxState[UnicycleD_x]):
    """Kinematic unicycle state: [x, y, heading]."""

    _array: StateArray

    @staticmethod
    def create(
        *,
        x: float | Scalar,
        y: float | Scalar,
        heading: float | Scalar,
    ) -> "JaxUnicycleState":
        return JaxUnicycleState(jnp.array([x, y, heading]))

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[UnicycleD_x]]:
        return np.asarray(self.array)

    @property
    def dimension(self) -> UnicycleD_x:
        return cast(UnicycleD_x, self.array.shape[0])

    @property
    def x(self) -> float:
        return float(self.array[0])

    @property
    def y(self) -> float:
        return float(self.array[1])

    @property
    def heading(self) -> float:
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
        return JaxUnicycleStateBatch(array)

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[JaxUnicycleState], *, horizon: T_ | None = None
    ) -> "JaxUnicycleStateBatch[T_, int]":
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
        return self._numpy_array

    def x(self) -> Array[Dims[T, M]]:
        return self._numpy_array[:, 0, :]

    def y(self) -> Array[Dims[T, M]]:
        return self._numpy_array[:, 1, :]

    @property
    def x_array(self) -> Float[JaxArray, "T M"]:
        return self.batch.array[:, 0, :]

    @property
    def y_array(self) -> Float[JaxArray, "T M"]:
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

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D[2], M]]:
        return np.asarray(self.batch.array[:, :2, :])


@jaxtyped
@dataclass(frozen=True)
class JaxUnicycleControlInputSequence[T: int](
    UnicycleControlInputSequence[T], JaxControlInputSequence[T, UnicycleD_u]
):
    """Control inputs: [linear velocity, angular velocity]."""

    _array: ControlInputSequenceArray[T]

    @staticmethod
    def zeroes[T_: int](horizon: T_) -> "JaxUnicycleControlInputSequence[T_]":
        return JaxUnicycleControlInputSequence(jnp.zeros((horizon, UNICYCLE_D_U)))

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, UnicycleD_u]]:
        return self._numpy_array

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

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, UnicycleD_u]]:
        return np.asarray(self.array)


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
        array = sequence.array[..., jnp.newaxis]

        assert array.shape == (expected := (sequence.horizon, UNICYCLE_D_U, 1)), (
            f"Array shape {array.shape} does not match expected shape {expected}."
        )

        return JaxUnicycleControlInputBatch(array)

    def __array__(
        self, dtype: DataType | None = None
    ) -> Array[Dims[T, UnicycleD_u, M]]:
        return self._numpy_array

    def linear_velocity(self) -> Array[Dims[T, M]]:
        return self._numpy_array[:, 0, :]

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

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, UnicycleD_u, M]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxUnicycleObstacleStates[K: int]:
    _array: Float[JaxArray, f"{UNICYCLE_D_O} K"]

    @staticmethod
    def wrap[K_: int](
        array: Array[Dims[UnicycleD_o, K_]] | Float[JaxArray, f"{UNICYCLE_D_O} K"],
    ) -> "JaxUnicycleObstacleStates[K_]":
        return JaxUnicycleObstacleStates(jnp.asarray(array))

    @staticmethod
    def create[K_: int](
        *,
        x: Array[Dims[K_]] | Float[JaxArray, "K_"],
        y: Array[Dims[K_]] | Float[JaxArray, "K_"],
        heading: Array[Dims[K_]] | Float[JaxArray, "K_"],
    ) -> "JaxUnicycleObstacleStates[K_]":
        return JaxUnicycleObstacleStates(jnp.stack([x, y, heading], axis=0))

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[UnicycleD_o, K]]:
        return self._numpy_array

    def x(self) -> Array[Dims[K]]:
        return self._numpy_array[0, :]

    def y(self) -> Array[Dims[K]]:
        return self._numpy_array[1, :]

    def heading(self) -> Array[Dims[K]]:
        return self._numpy_array[2, :]

    @property
    def dimension(self) -> UnicycleD_o:
        return cast(UnicycleD_o, self.array.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[1])

    @property
    def array(self) -> Float[JaxArray, f"{UNICYCLE_D_O} K"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[UnicycleD_o, K]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxUnicycleObstacleStateSequences[T: int, K: int]:
    _array: Float[JaxArray, f"T {UNICYCLE_ESTIMATION_D_X} K"]
    _covariance: Float[
        JaxArray, f"T {UNICYCLE_ESTIMATION_D_X} {UNICYCLE_ESTIMATION_D_X} K"
    ]

    @staticmethod
    def wrap[T_: int, K_: int](
        *,
        array: Float[JaxArray, f"T {UNICYCLE_ESTIMATION_D_X} K"],
        covariance: Float[
            JaxArray, f"T {UNICYCLE_ESTIMATION_D_X} {UNICYCLE_ESTIMATION_D_X} K"
        ],
        horizon: T_ | None = None,
        count: K_ | None = None,
    ) -> "JaxUnicycleObstacleStateSequences[T_, K_]":
        horizon = horizon if horizon is not None else cast(T_, array.shape[0])
        count = count if count is not None else cast(K_, array.shape[2])

        assert array.shape == (expected := (horizon, UNICYCLE_ESTIMATION_D_X, count)), (
            f"Array shape {array.shape} does not match expected shape {expected}."
        )
        assert covariance.shape == (
            expected := (
                horizon,
                UNICYCLE_ESTIMATION_D_X,
                UNICYCLE_ESTIMATION_D_X,
                count,
            )
        ), (
            f"Covariance shape {covariance.shape} does not match expected shape {expected}."
        )

        return JaxUnicycleObstacleStateSequences(array, covariance)

    def __array__(
        self, dtype: DataType | None = None
    ) -> Array[Dims[T, UnicycleEstimationD_x, K]]:
        return self._numpy_array

    def x(self) -> Array[Dims[T, K]]:
        return self._numpy_array[:, 0, :]

    def y(self) -> Array[Dims[T, K]]:
        return self._numpy_array[:, 1, :]

    def heading(self) -> Array[Dims[T, K]]:
        return self._numpy_array[:, 2, :]

    def covariance(
        self,
    ) -> Float[JaxArray, f"T {UNICYCLE_ESTIMATION_D_X} {UNICYCLE_ESTIMATION_D_X} K"]:
        return self._covariance

    def pose(self) -> JaxObstacle2dPoses[T, K]:
        return JaxObstacle2dPoses.create(
            x=self.x_array,
            y=self.y_array,
            heading=self.heading_array,
            covariance=self.pose_covariance_array,
        )

    @property
    def horizon(self) -> T:
        return cast(T, self._array.shape[0])

    @property
    def dimension(self) -> UnicycleEstimationD_x:
        return cast(UnicycleEstimationD_x, self._array.shape[1])

    @property
    def count(self) -> K:
        return cast(K, self._array.shape[2])

    @property
    def array(self) -> Float[JaxArray, f"T {UNICYCLE_ESTIMATION_D_X} K"]:
        return self._array

    @property
    def x_array(self) -> Float[JaxArray, "T K"]:
        return self._array[:, 0, :]

    @property
    def y_array(self) -> Float[JaxArray, "T K"]:
        return self._array[:, 1, :]

    @property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        return self._array[:, 2, :]

    @property
    def covariance_array(
        self,
    ) -> Float[JaxArray, f"T {UNICYCLE_ESTIMATION_D_X} {UNICYCLE_ESTIMATION_D_X} K"]:
        return self._covariance

    @property
    def pose_covariance_array(self) -> Float[JaxArray, f"T {POSE_D_O} {POSE_D_O} K"]:
        return self.covariance_array[:, :3, :3, :]

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, UnicycleEstimationD_x, K]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxUnicycleObstacleInputs[K: int]:
    _linear_velocities: Float[JaxArray, "K"]
    _angular_velocities: Float[JaxArray, "K"]

    @staticmethod
    def wrap[K_: int](
        array: Float[JaxArray, f"{UNICYCLE_D_U} K"] | Array[Dims[UnicycleD_u, K_]],
    ) -> "JaxUnicycleObstacleInputs[K_]":
        linear, angular = jnp.asarray(array)
        return JaxUnicycleObstacleInputs(
            _linear_velocities=linear, _angular_velocities=angular
        )

    @staticmethod
    def create(
        *,
        linear_velocities: Float[JaxArray, "K"] | Array[Dims[K]],
        angular_velocities: Float[JaxArray, "K"] | Array[Dims[K]],
    ) -> "JaxUnicycleObstacleInputs[int]":
        return JaxUnicycleObstacleInputs(
            _linear_velocities=jnp.asarray(linear_velocities),
            _angular_velocities=jnp.asarray(angular_velocities),
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[UnicycleD_u, K]]:
        return self._numpy_array

    def linear_velocities(self) -> Array[Dims[K]]:
        return self._numpy_array[0, :]

    def angular_velocities(self) -> Array[Dims[K]]:
        return self._numpy_array[1, :]

    def zeroed(
        self, *, linear_velocity: bool = False, angular_velocity: bool = False
    ) -> "JaxUnicycleObstacleInputs[K]":
        """Returns a version of the inputs with the specified components zeroed out."""
        return JaxUnicycleObstacleInputs(
            _linear_velocities=jnp.zeros_like(self.linear_velocities())
            if linear_velocity
            else self._linear_velocities,
            _angular_velocities=jnp.zeros_like(self.angular_velocities())
            if angular_velocity
            else self._angular_velocities,
        )

    @property
    def dimension(self) -> UnicycleD_u:
        return UNICYCLE_D_U

    @property
    def count(self) -> K:
        return cast(K, self._linear_velocities.shape[0])

    @property
    def linear_velocities_array(self) -> Float[JaxArray, "K"]:
        return self._linear_velocities

    @property
    def angular_velocities_array(self) -> Float[JaxArray, "K"]:
        return self._angular_velocities

    @property
    def array(self) -> Float[JaxArray, f"{UNICYCLE_D_U} K"]:
        return jnp.stack([self._linear_velocities, self._angular_velocities], axis=0)

    @cached_property
    def _numpy_array(self) -> Array[Dims[UnicycleD_u, K]]:
        return np.asarray(self.array)


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
    """Kinematic unicycle with direct velocity control, Euler-integrated with configurable limits."""

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


class JaxUnicycleStateEstimationModel(NamedTuple):
    """Kinematic unicycle model used for state estimation."""

    time_step_size: Scalar
    initial_state_covariance: EstimationStateCovarianceArray

    @staticmethod
    def create(
        *,
        time_step_size: float,
        initial_state_covariance: Array[
            Dims[UnicycleEstimationD_x, UnicycleEstimationD_x]
        ]
        | EstimationStateCovarianceArray
        | None = None,
    ) -> "JaxUnicycleStateEstimationModel":
        if initial_state_covariance is None:
            initial_state_covariance = (
                JaxUnicycleStateEstimationModel.default_initial_state_covariance()
            )

        return JaxUnicycleStateEstimationModel(
            time_step_size=jnp.asarray(time_step_size),
            initial_state_covariance=jnp.asarray(initial_state_covariance),
        )

    @staticmethod
    def default_initial_state_covariance() -> EstimationStateCovarianceArray:
        # NOTE: Sure of pose, unsure of velocities.
        return jnp.diag(
            jnp.array(
                [
                    SMALL_UNCERTAINTY,
                    SMALL_UNCERTAINTY,
                    SMALL_UNCERTAINTY,
                    LARGE_UNCERTAINTY,
                    LARGE_UNCERTAINTY,
                ]
            )
        )

    @jax.jit
    @jaxtyped
    def __call__(self, state: Float[JaxArray, "D_x K"]) -> Float[JaxArray, "D_x K"]:
        return jax.vmap(self.step, in_axes=1, out_axes=1)(state)

    @jax.jit
    @jaxtyped
    def jacobian(self, state: Float[JaxArray, "D_x K"]) -> Float[JaxArray, "D_x D_x K"]:
        return jax.vmap(jax.jacfwd(self.step))(state.T).transpose(1, 2, 0)

    @jax.jit
    @jaxtyped
    def step(self, state: Float[JaxArray, "D_x"]) -> Float[JaxArray, "D_x"]:
        dt = self.time_step_size
        x, y, theta, v, omega = state
        return jnp.array(
            [
                x + v * jnp.cos(theta) * dt,
                y + v * jnp.sin(theta) * dt,
                theta + omega * dt,
                v,
                omega,
            ]
        )

    def observations_from[T: int = int, K: int = int](
        self, history: JaxUnicycleObstacleStatesHistory[T, K]
    ) -> Float[JaxArray, "T D_z K"]:
        return jnp.stack(
            [history.x_array, history.y_array, history.heading_array], axis=1
        )

    def states_from[K: int = int](
        self, belief: UnicycleGaussianBelief[K]
    ) -> JaxUnicycleObstacleStates[K]:
        return JaxUnicycleObstacleStates.wrap(belief.mean[:3, :])

    def inputs_from[K: int = int](
        self, belief: UnicycleGaussianBelief[K]
    ) -> JaxUnicycleObstacleInputs[K]:
        return cast(
            JaxUnicycleObstacleInputs[K],
            JaxUnicycleObstacleInputs.wrap(belief.mean[3:, :]),
        )

    def initial_belief_from[K: int](
        self,
        *,
        states: JaxUnicycleObstacleStates[K],
        inputs: JaxUnicycleObstacleInputs[K],
        covariances: JaxUnicycleObstacleCovariances[K] | None = None,
    ) -> UnicycleGaussianBelief[K]:
        augmented = jnp.concatenate([states.array, inputs.array], axis=0)

        if covariances is None:
            # NOTE: No covariance means we are "certain" about the states.
            covariances = jnp.broadcast_to(
                jnp.eye(UNICYCLE_ESTIMATION_D_X)[:, :, jnp.newaxis] * SMALL_UNCERTAINTY,
                (UNICYCLE_ESTIMATION_D_X, UNICYCLE_ESTIMATION_D_X, states.count),
            )

        return JaxGaussianBelief(mean=augmented, covariance=covariances)

    @property
    def observation_matrix(self) -> ObservationMatrix:
        # NOTE: We observe the pose directly.
        return jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )


@dataclass(kw_only=True, frozen=True)
class JaxUnicycleObstacleModel(
    ObstacleModel[
        JaxUnicycleObstacleStatesHistory,
        JaxUnicycleObstacleStates,
        JaxUnicycleObstacleInputs,
        JaxUnicycleObstacleCovariances,
        JaxUnicycleObstacleStateSequences,
    ]
):
    """Propagates unicycle kinematics forward given states and control inputs."""

    model: "JaxUnicycleStateEstimationModel"
    process_noise_covariance: ProcessNoiseCovarianceArray
    predictor: KalmanFilter

    @staticmethod
    def unscented(
        *,
        time_step_size: float,
        process_noise_covariance: JaxNoiseCovarianceDescription = 1e-3,
        sigma_point_spread: float = 1.0,
        prior_knowledge: float = 2.0,
    ) -> "JaxUnicycleObstacleModel":
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
        standardized_covariance = jax_kalman_filter.standardize_noise_covariance(
            process_noise_covariance, dimension=UNICYCLE_ESTIMATION_D_X
        )
        return JaxUnicycleObstacleModel(
            model=JaxUnicycleStateEstimationModel.create(
                time_step_size=time_step_size,
            ),
            process_noise_covariance=standardized_covariance,
            predictor=JaxUnscentedKalmanFilter.create(
                alpha=sigma_point_spread, beta=prior_knowledge
            ),
        )

    def forward[T: int, K: int](
        self,
        *,
        states: JaxUnicycleObstacleStates[K],
        inputs: JaxUnicycleObstacleInputs[K],
        covariances: JaxUnicycleObstacleCovariances[K] | None,
        horizon: T,
    ) -> JaxUnicycleObstacleStateSequences[T, K]:
        means, covariance = forward(
            initial=self.model.initial_belief_from(
                states=states, inputs=inputs, covariances=covariances
            ),
            model=self.model,
            predictor=self.predictor,
            process_noise_covariance=self.process_noise_covariance,
            horizon=horizon,
        )

        return JaxUnicycleObstacleStateSequences.wrap(
            array=means, covariance=covariance
        )


@dataclass(frozen=True)
class JaxFiniteDifferenceUnicycleStateEstimator(
    ObstacleStateEstimator[
        JaxUnicycleObstacleStatesHistory,
        JaxUnicycleObstacleStates,
        JaxUnicycleObstacleInputs,
    ]
):
    time_step_size: Scalar

    @staticmethod
    def create(*, time_step_size: float) -> "JaxFiniteDifferenceUnicycleStateEstimator":
        return JaxFiniteDifferenceUnicycleStateEstimator(
            time_step_size=jnp.asarray(time_step_size)
        )

    def estimate_from[K: int](
        self, history: JaxUnicycleObstacleStatesHistory[int, K]
    ) -> EstimatedObstacleStates[
        JaxUnicycleObstacleStates[K], JaxUnicycleObstacleInputs[K], None
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

        estimated = estimate_states(
            x_history=history.x_array,
            y_history=history.y_array,
            heading_history=history.heading_array,
            time_step_size=self.time_step_size,
        )

        return EstimatedObstacleStates(
            states=JaxUnicycleObstacleStates.create(
                x=estimated.x,
                y=estimated.y,
                heading=estimated.heading,
            ),
            inputs=cast(
                JaxUnicycleObstacleInputs[K],
                JaxUnicycleObstacleInputs.create(
                    linear_velocities=estimated.linear_velocities,
                    angular_velocities=estimated.angular_velocities,
                ),
            ),
            covariance=None,
        )


@dataclass(frozen=True)
class JaxKfUnicycleStateEstimator(
    ObstacleStateEstimator[
        JaxUnicycleObstacleStatesHistory,
        JaxUnicycleObstacleStates,
        JaxUnicycleObstacleInputs,
        JaxUnicycleObstacleCovariances,
    ]
):
    """Kalman Filter state estimator for unicycle model obstacles."""

    process_noise_covariance: ProcessNoiseCovarianceArray
    observation_noise_covariance: ObservationNoiseCovarianceArray
    model: JaxUnicycleStateEstimationModel
    estimator: KalmanFilter

    @staticmethod
    def ekf(
        *,
        time_step_size: float,
        process_noise_covariance: JaxNoiseCovarianceDescription[UnicycleEstimationD_x],
        observation_noise_covariance: JaxNoiseCovarianceDescription[
            UnicycleObservationD_o
        ],
        initial_state_covariance: Array[
            Dims[UnicycleEstimationD_x, UnicycleEstimationD_x]
        ]
        | EstimationStateCovarianceArray
        | None = None,
    ) -> "JaxKfUnicycleStateEstimator":
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
        return JaxKfUnicycleStateEstimator(
            process_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                process_noise_covariance, dimension=UNICYCLE_ESTIMATION_D_X
            ),
            observation_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=UNICYCLE_OBSERVATION_D_O
            ),
            model=JaxUnicycleStateEstimationModel.create(
                time_step_size=time_step_size,
                initial_state_covariance=initial_state_covariance,
            ),
            estimator=JaxExtendedKalmanFilter.create(),
        )

    @staticmethod
    def ukf(
        *,
        time_step_size: float,
        process_noise_covariance: JaxNoiseCovarianceDescription[UnicycleEstimationD_x],
        observation_noise_covariance: JaxNoiseCovarianceDescription[
            UnicycleObservationD_o
        ],
        initial_state_covariance: Array[
            Dims[UnicycleEstimationD_x, UnicycleEstimationD_x]
        ]
        | EstimationStateCovarianceArray
        | None = None,
    ) -> "JaxKfUnicycleStateEstimator":
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
        return JaxKfUnicycleStateEstimator(
            process_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                process_noise_covariance, dimension=UNICYCLE_ESTIMATION_D_X
            ),
            observation_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=UNICYCLE_OBSERVATION_D_O
            ),
            model=JaxUnicycleStateEstimationModel.create(
                time_step_size=time_step_size,
                initial_state_covariance=initial_state_covariance,
            ),
            estimator=JaxUnscentedKalmanFilter.create(),
        )

    def estimate_from[K: int, T: int = int](
        self, history: JaxUnicycleObstacleStatesHistory[T, K]
    ) -> EstimatedObstacleStates[
        JaxUnicycleObstacleStates[K],
        JaxUnicycleObstacleInputs[K],
        JaxUnicycleObstacleCovariances[K],
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


class EstimatedUnicycleObstacleStates(NamedTuple):
    x: Float[JaxArray, "K"]
    y: Float[JaxArray, "K"]
    heading: Float[JaxArray, "K"]
    linear_velocities: Float[JaxArray, "K"]
    angular_velocities: Float[JaxArray, "K"]


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


@jax.jit
@jaxtyped
def estimate_states(
    *,
    x_history: Float[JaxArray, "T K"],
    y_history: Float[JaxArray, "T K"],
    heading_history: Float[JaxArray, "T K"],
    time_step_size: Scalar,
) -> EstimatedUnicycleObstacleStates:
    filter_invalid = invalid_obstacle_filter_from(
        x_history, y_history, heading_history, check_recent=2
    )

    return EstimatedUnicycleObstacleStates(
        x=x_history[-1],
        y=y_history[-1],
        heading=heading_history[-1],
        linear_velocities=filter_invalid(
            estimate_linear_velocities(
                x_history=x_history,
                y_history=y_history,
                heading_history=heading_history,
                time_step_size=time_step_size,
            )
        ),
        angular_velocities=filter_invalid(
            estimate_angular_velocities(
                heading_history=heading_history,
                time_step_size=time_step_size,
            )
        ),
    )


@jax.jit
@jaxtyped
def estimate_linear_velocities(
    *,
    x_history: Float[JaxArray, "T K"],
    y_history: Float[JaxArray, "T K"],
    heading_history: Float[JaxArray, "T K"],
    time_step_size: Scalar,
) -> Float[JaxArray, "K"]:
    horizon = x_history.shape[0]
    obstacle_count = x_history.shape[1]

    def estimate(
        *,
        x_current: Float[JaxArray, "K"],
        y_current: Float[JaxArray, "K"],
        x_previous: Float[JaxArray, "K"],
        y_previous: Float[JaxArray, "K"],
        heading: Float[JaxArray, "K"],
    ) -> Float[JaxArray, "K"]:
        delta_x = x_current - x_previous
        delta_y = y_current - y_previous

        return (
            delta_x * jnp.cos(heading) + delta_y * jnp.sin(heading)
        ) / time_step_size

    if horizon < 2:
        return jnp.zeros(obstacle_count)

    return estimate(
        x_current=x_history[-1],
        y_current=y_history[-1],
        x_previous=x_history[-2],
        y_previous=y_history[-2],
        heading=heading_history[-1],
    )


@jax.jit
@jaxtyped
def estimate_angular_velocities(
    *,
    heading_history: Float[JaxArray, "T K"],
    time_step_size: Scalar,
) -> Float[JaxArray, "K"]:
    horizon = heading_history.shape[0]
    obstacle_count = heading_history.shape[1]

    def estimate(
        *, heading_current: Float[JaxArray, "K"], heading_previous: Float[JaxArray, "K"]
    ) -> Float[JaxArray, "K"]:
        return (heading_current - heading_previous) / time_step_size

    if horizon < 2:
        return jnp.zeros(obstacle_count)

    return estimate(
        heading_current=heading_history[-1], heading_previous=heading_history[-2]
    )


@jax.jit(static_argnames=("horizon",))
@jaxtyped
def forward(
    *,
    initial: UnicycleGaussianBelief,
    model: JaxUnicycleStateEstimationModel,
    predictor: KalmanFilter,
    process_noise_covariance: ProcessNoiseCovarianceArray,
    horizon: int,
) -> tuple[
    Float[JaxArray, f"T {UNICYCLE_ESTIMATION_D_X} K"],
    Float[JaxArray, f"T {UNICYCLE_ESTIMATION_D_X} {UNICYCLE_ESTIMATION_D_X} K"],
]:
    def step(
        belief: UnicycleGaussianBelief, _
    ) -> tuple[UnicycleGaussianBelief, UnicycleGaussianBelief]:
        next_belief = predictor.predict(
            belief=belief,
            state_transition=model,
            process_noise_covariance=process_noise_covariance,
        )

        return next_belief, next_belief

    _, beliefs = jax.lax.scan(step, initial, xs=None, length=horizon)

    return beliefs.mean, beliefs.covariance
