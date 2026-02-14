from typing import Never, cast, overload, Self, Sequence, Final, Any, NamedTuple
from dataclasses import dataclass
from functools import cached_property

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
    BicycleD_o,
    BICYCLE_D_O,
    BICYCLE_POSE_D_O,
    BICYCLE_POSITION_D_O,
    DynamicalModel,
    ObstacleModel,
    ObstacleStateEstimator,
    EstimatedObstacleStates,
    CovarianceExtractor,
)

from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims, D, shape_of

from trajax.filters import (
    JaxExtendedKalmanFilter,
    JaxGaussianBelief,
    JaxNoiseCovarianceDescription,
    JaxUnscentedKalmanFilter,
    jax_kalman_filter,
)

import jax
import jax.numpy as jnp
import numpy as np


NO_LIMITS: Final = (jnp.asarray(-jnp.inf), jnp.asarray(jnp.inf))

BICYCLE_ESTIMATION_D_X: Final = 6
BICYCLE_OBSERVATION_D_O: Final = 3

type BicycleEstimationD_x = D[6]
type BicycleObservationD_o = D[3]

type StateArray = Float[JaxArray, f"{BICYCLE_D_X}"]
type ControlInputSequenceArray[T: int] = Float[JaxArray, f"T {BICYCLE_D_U}"]
type StateBatchArray[T: int, M: int] = Float[JaxArray, f"T {BICYCLE_D_X} M"]
type ControlInputBatchArray[T: int, M: int] = Float[JaxArray, f"T {BICYCLE_D_U} M"]
type EstimationStateCovarianceArray = Float[
    JaxArray, f"{BICYCLE_ESTIMATION_D_X} {BICYCLE_ESTIMATION_D_X}"
]
type ProcessNoiseCovarianceArray = Float[
    JaxArray, f"{BICYCLE_ESTIMATION_D_X} {BICYCLE_ESTIMATION_D_X}"
]
type ObservationNoiseCovarianceArray = Float[
    JaxArray, f"{BICYCLE_OBSERVATION_D_O} {BICYCLE_OBSERVATION_D_O}"
]
type ObservationMatrix = Float[
    JaxArray, f"{BICYCLE_OBSERVATION_D_O} {BICYCLE_ESTIMATION_D_X}"
]

type StatesAtTimeStep[M: int] = Float[JaxArray, f"{BICYCLE_D_X} M"]
type ControlInputsAtTimeStep[M: int] = Float[JaxArray, f"{BICYCLE_D_U} M"]

type BicycleObstacleStateCovarianceArray[T: int, K: int] = Float[
    JaxArray, f"T {BICYCLE_D_O} {BICYCLE_D_O} K"
]
type BicycleObstaclePoseCovarianceArray[T: int, K: int] = Float[
    JaxArray, f"T {BICYCLE_POSE_D_O} {BICYCLE_POSE_D_O} K"
]
type BicycleObstaclePositionCovarianceArray[T: int, K: int] = Float[
    JaxArray, f"T {BICYCLE_POSITION_D_O} {BICYCLE_POSITION_D_O} K"
]

type KalmanFilter = JaxExtendedKalmanFilter | JaxUnscentedKalmanFilter


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleState(BicycleState, JaxState[BicycleD_x]):
    """Kinematic bicycle state: [x, y, heading, speed]."""

    _array: StateArray

    @staticmethod
    def create(
        *,
        x: float | Scalar,
        y: float | Scalar,
        heading: float | Scalar,
        speed: float | Scalar,
    ) -> "JaxBicycleState":
        return JaxBicycleState(jnp.array([x, y, heading, speed]))

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[BicycleD_x]]:
        return np.asarray(self.array)

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
    def heading(self) -> float:
        return float(self.array[2])

    @property
    def speed(self) -> float:
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
        return JaxBicycleStateBatch(array)

    @staticmethod
    def of_states[T_: int = int](
        states: Sequence[JaxBicycleState], *, horizon: T_ | None = None
    ) -> "JaxBicycleStateBatch[T_, int]":
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
    """Control inputs: [acceleration, steering_angle]."""

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
    array: Float[JaxArray, f"{BICYCLE_D_O} K"]
    _covariance: Float[JaxArray, f"{BICYCLE_D_O} {BICYCLE_D_O} K"] | None = None
    """State covariance matrix from Kalman filtering. Shape: (BicycleD_o, BicycleD_o, K)."""

    @staticmethod
    def wrap[K_: int](
        array: Array[Dims[BicycleD_o, K_]] | Float[JaxArray, f"{BICYCLE_D_O} K"],
        *,
        covariance: Float[JaxArray, f"{BICYCLE_D_O} {BICYCLE_D_O} K"] | None = None,
    ) -> "JaxBicycleObstacleStates[K_]":
        return JaxBicycleObstacleStates(jnp.asarray(array), covariance)

    @staticmethod
    def create[K_: int](
        *,
        x: Array[Dims[K_]] | Float[JaxArray, "K_"],
        y: Array[Dims[K_]] | Float[JaxArray, "K_"],
        heading: Array[Dims[K_]] | Float[JaxArray, "K_"],
        speed: Array[Dims[K_]] | Float[JaxArray, "K_"],
        count: K_ | None = None,
        covariance: Float[JaxArray, f"{BICYCLE_D_O} {BICYCLE_D_O} K"] | None = None,
    ) -> "JaxBicycleObstacleStates[K_]":
        array = jnp.stack([x, y, heading, speed], axis=0)
        return JaxBicycleObstacleStates(array, covariance)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[BicycleD_o, K]]:
        return self._numpy_array

    def x(self) -> Array[Dims[K]]:
        return self._numpy_array[0, :]

    def y(self) -> Array[Dims[K]]:
        return self._numpy_array[1, :]

    def heading(self) -> Array[Dims[K]]:
        return self._numpy_array[2, :]

    def speed(self) -> Array[Dims[K]]:
        return self._numpy_array[3, :]

    @property
    def dimension(self) -> BicycleD_o:
        return cast(BicycleD_o, self.array.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[1])

    @property
    def covariance(self) -> Float[JaxArray, f"{BICYCLE_D_O} {BICYCLE_D_O} K"] | None:
        return self._covariance

    @cached_property
    def _numpy_array(self) -> Array[Dims[BicycleD_o, K]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleObstacleStateSequences[T: int, K: int]:
    array: Float[JaxArray, f"T {BICYCLE_D_O} K"]

    @staticmethod
    def create(
        *,
        x: Float[JaxArray, "T K"],
        y: Float[JaxArray, "T K"],
        heading: Float[JaxArray, "T K"],
        speed: Float[JaxArray, "T K"],
    ) -> "JaxBicycleObstacleStateSequences[int, int]":
        array = jnp.stack([x, y, heading, speed], axis=1)
        return JaxBicycleObstacleStateSequences(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_x, K]]:
        return self._numpy_array

    def single(self) -> Never:
        # TODO: Fix this!
        raise NotImplementedError(
            "single() is not implemented for JaxBicycleObstacleStateSequences."
        )

    def x(self) -> Array[Dims[T, K]]:
        return self._numpy_array[:, 0, :]

    def y(self) -> Array[Dims[T, K]]:
        return self._numpy_array[:, 1, :]

    def heading(self) -> Array[Dims[T, K]]:
        return self._numpy_array[:, 2, :]

    def speed(self) -> Array[Dims[T, K]]:
        return self._numpy_array[:, 3, :]

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

    @property
    def speed_array(self) -> Float[JaxArray, "T K"]:
        return self.array[:, 3, :]

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, BicycleD_x, K]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleObstacleInputs[K: int]:
    _accelerations: Float[JaxArray, "K"]
    _steering_angles: Float[JaxArray, "K"]
    _covariance: Float[JaxArray, f"{BICYCLE_D_U} {BICYCLE_D_U} K"] | None = None
    """Input covariance matrix from Kalman filtering. Shape: (BicycleD_u, BicycleD_u, K)."""

    @staticmethod
    def wrap[K_: int](
        inputs: Array[Dims[BicycleD_u, K_]] | Float[JaxArray, f"{BICYCLE_D_U} K"],
        *,
        covariance: Float[JaxArray, f"{BICYCLE_D_U} {BICYCLE_D_U} K"] | None = None,
    ) -> "JaxBicycleObstacleInputs[K_]":
        inputs = jnp.asarray(inputs)
        return JaxBicycleObstacleInputs(
            _accelerations=inputs[0], _steering_angles=inputs[1], _covariance=covariance
        )

    @staticmethod
    def create[K_: int](
        *,
        accelerations: Array[Dims[K_]] | Float[JaxArray, "K"],
        steering_angles: Array[Dims[K_]] | Float[JaxArray, "K"],
        covariance: Float[JaxArray, f"{BICYCLE_D_U} {BICYCLE_D_U} K"] | None = None,
    ) -> "JaxBicycleObstacleInputs[int]":
        return JaxBicycleObstacleInputs(
            _accelerations=jnp.asarray(accelerations),
            _steering_angles=jnp.asarray(steering_angles),
            _covariance=covariance,
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[BicycleD_u, K]]:
        return self._numpy_array

    def zeroed(
        self, *, acceleration: bool = False, steering_angle: bool = False
    ) -> "JaxBicycleObstacleInputs[K]":
        """Returns a version of the inputs with the specified components zeroed out."""
        return JaxBicycleObstacleInputs(
            _accelerations=jnp.zeros_like(self._accelerations)
            if acceleration
            else self._accelerations,
            _steering_angles=jnp.zeros_like(self._steering_angles)
            if steering_angle
            else self._steering_angles,
            _covariance=self._covariance,
        )

    def accelerations(self) -> Array[Dims[K]]:
        return self._numpy_array[0, :]

    def steering_angles(self) -> Array[Dims[K]]:
        return self._numpy_array[1, :]

    @property
    def dimension(self) -> BicycleD_u:
        return BICYCLE_D_U

    @property
    def count(self) -> K:
        return cast(K, self._steering_angles.shape[0])

    @property
    def covariance(self) -> Float[JaxArray, f"{BICYCLE_D_U} {BICYCLE_D_U} K"] | None:
        return self._covariance

    @property
    def accelerations_array(self) -> Float[JaxArray, "K"]:
        return self._accelerations

    @property
    def steering_angles_array(self) -> Float[JaxArray, "K"]:
        return self._steering_angles

    @cached_property
    def _numpy_array(self) -> Array[Dims[BicycleD_u, K]]:
        array = np.stack(
            [np.asarray(self._accelerations), np.asarray(self._steering_angles)], axis=0
        )

        assert shape_of(array, matches=(BICYCLE_D_U, self.count))

        return array


@jaxtyped
@dataclass(frozen=True)
class JaxBicycleObstacleControlInputSequences[T: int, K: int]:
    array: Float[JaxArray, f"T {BICYCLE_D_U} K"]

    @staticmethod
    def wrap[T_: int, K_: int](
        array: Array[Dims[T_, BicycleD_u, K_]] | Float[JaxArray, f"{BICYCLE_D_U} K"],
    ) -> "JaxBicycleObstacleControlInputSequences[T_, K_]":
        return JaxBicycleObstacleControlInputSequences(jnp.asarray(array))

    @staticmethod
    def create[T_: int, K_: int](
        *,
        accelerations: Array[Dims[T_, K_]] | Float[JaxArray, "T K"],
        steering_angles: Array[Dims[T_, K_]] | Float[JaxArray, "T K"],
    ) -> "JaxBicycleObstacleControlInputSequences[T_, K_]":
        array = jnp.stack([accelerations, steering_angles], axis=1)
        return JaxBicycleObstacleControlInputSequences(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_u, K]]:
        return self._numpy_array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> BicycleD_u:
        return cast(BicycleD_u, self.array.shape[1])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[2])

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, BicycleD_u, K]]:
        return np.asarray(self.array)


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
    """Kinematic bicycle with rear-axle reference, Euler-integrated with configurable limits."""

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
        JaxBicycleObstacleInputs,
        JaxBicycleObstacleStateSequences,
    ]
):
    """Propagates bicycle kinematics forward given states and control inputs."""

    time_step_size: Scalar
    wheelbase: Scalar

    @staticmethod
    def create(
        *, time_step_size: float, wheelbase: float = 1.0
    ) -> "JaxBicycleObstacleModel":
        return JaxBicycleObstacleModel(
            time_step_size=jnp.asarray(time_step_size),
            wheelbase=jnp.asarray(wheelbase),
        )

    def forward[T: int, K: int](
        self,
        *,
        current: JaxBicycleObstacleStates[K],
        inputs: JaxBicycleObstacleInputs[K],
        horizon: T,
    ) -> JaxBicycleObstacleStateSequences[T, K]:
        input_sequences = self._input_to_maintain(inputs, horizon=horizon)

        return JaxBicycleObstacleStateSequences(
            simulate(
                input_sequences.array,
                current.array,
                time_step_size=self.time_step_size,
                wheelbase=self.wheelbase,
                speed_limits=NO_LIMITS,
                steering_limits=NO_LIMITS,
                acceleration_limits=NO_LIMITS,
            )
        )

    def state_jacobian[T: int, K: int](
        self,
        *,
        states: JaxBicycleObstacleStateSequences[T, K],
        inputs: JaxBicycleObstacleInputs[K],
    ) -> Float[JaxArray, "T 4 4 K"]:
        input_sequences = self._input_to_maintain(inputs, horizon=states.horizon)

        return state_jacobian(
            states.array,
            input_sequences.array,
            time_step_size=self.time_step_size,
            wheelbase=self.wheelbase,
        )

    def input_jacobian[T: int, K: int](
        self,
        *,
        states: JaxBicycleObstacleStateSequences[T, K],
        inputs: JaxBicycleObstacleInputs[K],
    ) -> Float[JaxArray, "T 4 2 K"]:
        input_sequences = self._input_to_maintain(inputs, horizon=states.horizon)

        return input_jacobian(
            states.array,
            input_sequences.array,
            time_step_size=self.time_step_size,
            wheelbase=self.wheelbase,
        )

    def _input_to_maintain[T: int, K: int](
        self, inputs: JaxBicycleObstacleInputs[K], *, horizon: T
    ) -> JaxBicycleObstacleControlInputSequences[T, K]:
        return JaxBicycleObstacleControlInputSequences.create(
            accelerations=jnp.tile(
                inputs.accelerations_array[jnp.newaxis, :], (horizon, 1)
            ),
            steering_angles=jnp.tile(
                inputs.steering_angles_array[jnp.newaxis, :], (horizon, 1)
            ),
        )


class JaxBicycleStateEstimationModel(NamedTuple):
    """Kinematic bicycle model used for state estimation."""

    time_step_size: Scalar
    wheelbase: Scalar

    @staticmethod
    def create(
        *, time_step_size: float, wheelbase: float
    ) -> "JaxBicycleStateEstimationModel":
        return JaxBicycleStateEstimationModel(
            time_step_size=jnp.asarray(time_step_size),
            wheelbase=jnp.asarray(wheelbase),
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
        x, y, theta, v, a, delta = state
        return jnp.array(
            [
                x + v * jnp.cos(theta) * dt,
                y + v * jnp.sin(theta) * dt,
                theta + (v / self.wheelbase) * jnp.tan(delta) * dt,
                v + a * dt,
                a,
                delta,
            ]
        )

    def observations_from(
        self, history: "JaxBicycleObstacleStatesHistory"
    ) -> Float[JaxArray, "T D_z K"]:
        return jnp.stack(
            [history.x_array, history.y_array, history.heading_array], axis=1
        )

    def x(self, belief: JaxGaussianBelief) -> Float[JaxArray, "K"]:
        return belief.mean[0, :]

    def y(self, belief: JaxGaussianBelief) -> Float[JaxArray, "K"]:
        return belief.mean[1, :]

    def heading(self, belief: JaxGaussianBelief) -> Float[JaxArray, "K"]:
        return belief.mean[2, :]

    def speed(self, belief: JaxGaussianBelief) -> Float[JaxArray, "K"]:
        return belief.mean[3, :]

    def acceleration(self, belief: JaxGaussianBelief) -> Float[JaxArray, "K"]:
        return belief.mean[4, :]

    def steering_angle(self, belief: JaxGaussianBelief) -> Float[JaxArray, "K"]:
        return belief.mean[5, :]

    @property
    def initial_state_covariance(self) -> EstimationStateCovarianceArray:
        # NOTE: Sure of pose, unsure of everything else.
        # Steering angle covariance is kept small (0.1) to prevent UKF sigma point
        # spread from reaching regions where tan(delta) becomes highly nonlinear.
        return jnp.diag(jnp.array([1e-4, 1e-4, 1e-4, 1.0, 1.0, 0.1]))

    @property
    def observation_matrix(self) -> ObservationMatrix:
        # NOTE: We observe the pose directly.
        return jnp.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )


@dataclass(frozen=True)
class JaxFiniteDifferenceBicycleStateEstimator(
    ObstacleStateEstimator[
        JaxBicycleObstacleStatesHistory,
        JaxBicycleObstacleStates,
        JaxBicycleObstacleInputs,
    ]
):
    time_step_size: Scalar
    wheelbase: Scalar

    @staticmethod
    def create(
        *, time_step_size: float, wheelbase: float
    ) -> "JaxFiniteDifferenceBicycleStateEstimator":
        return JaxFiniteDifferenceBicycleStateEstimator(
            time_step_size=jnp.asarray(time_step_size), wheelbase=jnp.asarray(wheelbase)
        )

    def estimate_from[K: int, T: int = int](
        self, history: JaxBicycleObstacleStatesHistory[T, K]
    ) -> EstimatedObstacleStates[
        JaxBicycleObstacleStates[K], JaxBicycleObstacleInputs[K], None
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

        estimated = estimate_states(
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
                speed=estimated.speed,
            ),
            inputs=cast(
                JaxBicycleObstacleInputs[K],
                JaxBicycleObstacleInputs.create(
                    accelerations=estimated.accelerations,
                    steering_angles=estimated.steering_angles,
                ),
            ),
            covariance=None,
        )


@dataclass(frozen=True)
class JaxKfBicycleStateEstimator(
    ObstacleStateEstimator[
        JaxBicycleObstacleStatesHistory,
        JaxBicycleObstacleStates,
        JaxBicycleObstacleInputs,
    ]
):
    """Kalman Filter state estimator for bicycle model obstacles."""

    process_noise_covariance: ProcessNoiseCovarianceArray
    observation_noise_covariance: ObservationNoiseCovarianceArray
    model: JaxBicycleStateEstimationModel
    estimator: KalmanFilter

    @staticmethod
    def ekf(
        *,
        time_step_size: float,
        wheelbase: float,
        process_noise_covariance: JaxNoiseCovarianceDescription[BicycleEstimationD_x],
        observation_noise_covariance: JaxNoiseCovarianceDescription[
            BicycleObservationD_o
        ],
    ) -> "JaxKfBicycleStateEstimator":
        """Creates an EKF state estimator for the bicycle model with the specified noise
        covariances.

        Args:
            time_step_size: The time step size for the state transition model.
            wheelbase: The wheelbase of the bicycle.
            process_noise_covariance: The process noise covariance, either as a full
                matrix, a vector of diagonal entries, or a scalar for isotropic noise.
            observation_noise_covariance: The observation noise covariance, either as a
                full matrix, a vector of diagonal entries, or a scalar for isotropic noise.
        """
        return JaxKfBicycleStateEstimator(
            process_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                process_noise_covariance, dimension=BICYCLE_ESTIMATION_D_X
            ),
            observation_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=BICYCLE_OBSERVATION_D_O
            ),
            model=JaxBicycleStateEstimationModel.create(
                time_step_size=time_step_size, wheelbase=wheelbase
            ),
            estimator=JaxExtendedKalmanFilter.create(),
        )

    @staticmethod
    def ukf(
        *,
        time_step_size: float,
        wheelbase: float,
        process_noise_covariance: JaxNoiseCovarianceDescription[BicycleEstimationD_x],
        observation_noise_covariance: JaxNoiseCovarianceDescription[
            BicycleObservationD_o
        ],
    ) -> "JaxKfBicycleStateEstimator":
        """Creates a UKF state estimator for the bicycle model with the specified noise
        covariances.

        Args:
            time_step_size: The time step size for the state transition model.
            wheelbase: The wheelbase of the bicycle.
            process_noise_covariance: The process noise covariance, either as a full
                matrix, a vector of diagonal entries, or a scalar for isotropic noise.
            observation_noise_covariance: The observation noise covariance, either as a
                full matrix, a vector of diagonal entries, or a scalar for isotropic noise.
        """
        return JaxKfBicycleStateEstimator(
            process_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                process_noise_covariance, dimension=BICYCLE_ESTIMATION_D_X
            ),
            observation_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=BICYCLE_OBSERVATION_D_O
            ),
            model=JaxBicycleStateEstimationModel.create(
                time_step_size=time_step_size, wheelbase=wheelbase
            ),
            estimator=JaxUnscentedKalmanFilter.create(),
        )

    def estimate_from[K: int, T: int = int](
        self, history: JaxBicycleObstacleStatesHistory[T, K]
    ) -> EstimatedObstacleStates[
        JaxBicycleObstacleStates[K],
        JaxBicycleObstacleInputs[K],
        Float[JaxArray, "D_x D_x K"],
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
            states=self.states_from(estimate),
            inputs=self.inputs_from(estimate),
            covariance=estimate.covariance,
        )

    def states_from[K: int = int](
        self, belief: JaxGaussianBelief
    ) -> JaxBicycleObstacleStates[K]:
        return JaxBicycleObstacleStates.create(
            x=self.model.x(belief),
            y=self.model.y(belief),
            heading=self.model.heading(belief),
            speed=self.model.speed(belief),
        )

    def inputs_from[K: int = int](
        self, belief: JaxGaussianBelief
    ) -> JaxBicycleObstacleInputs[K]:
        return cast(
            JaxBicycleObstacleInputs[K],
            JaxBicycleObstacleInputs.create(
                accelerations=self.model.acceleration(belief),
                steering_angles=self.model.steering_angle(belief),
            ),
        )


class JaxBicyclePoseCovarianceExtractor(CovarianceExtractor):
    """Extracts the pose-related state covariance from the full state covariance for bicycle obstacles."""

    def __call__[T: int, K: int](
        self, covariance: BicycleObstacleStateCovarianceArray[T, K]
    ) -> BicycleObstaclePoseCovarianceArray[T, K]:
        return covariance[:, :BICYCLE_POSE_D_O, :BICYCLE_POSE_D_O, :]


class JaxBicyclePositionCovarianceExtractor(CovarianceExtractor):
    """Extracts the x and y covariance from the full state covariance for bicycle obstacles."""

    def __call__[T: int, K: int](
        self, covariance: BicycleObstacleStateCovarianceArray[T, K]
    ) -> BicycleObstaclePositionCovarianceArray[T, K]:
        return covariance[:, :BICYCLE_POSITION_D_O, :BICYCLE_POSITION_D_O, :]


class EstimatedBicycleObstacleStates(NamedTuple):
    speed: Float[JaxArray, "K"]
    accelerations: Float[JaxArray, "K"]
    steering_angles: Float[JaxArray, "K"]


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
def state_jacobian(
    states: Float[JaxArray, "T 4 K"],
    controls: Float[JaxArray, "T 2 K"],
    *,
    time_step_size: Scalar,
    wheelbase: Scalar,
) -> Float[JaxArray, "T 4 4 K"]:
    theta = states[:, 2, :]
    v = states[:, 3, :]
    delta = controls[:, 1, :]

    T, K = theta.shape
    dt = time_step_size

    F = jnp.zeros((T, BICYCLE_D_X, BICYCLE_D_X, K))
    F = F.at[:, 0, 0, :].set(1.0)
    F = F.at[:, 1, 1, :].set(1.0)
    F = F.at[:, 2, 2, :].set(1.0)
    F = F.at[:, 3, 3, :].set(1.0)

    F = F.at[:, 0, 2, :].set(-v * jnp.sin(theta) * dt)
    F = F.at[:, 0, 3, :].set(jnp.cos(theta) * dt)
    F = F.at[:, 1, 2, :].set(v * jnp.cos(theta) * dt)
    F = F.at[:, 1, 3, :].set(jnp.sin(theta) * dt)
    F = F.at[:, 2, 3, :].set(jnp.tan(delta) / wheelbase * dt)

    return F


@jax.jit
@jaxtyped
def input_jacobian(
    states: Float[JaxArray, "T 4 K"],
    controls: Float[JaxArray, "T 2 K"],
    *,
    time_step_size: Scalar,
    wheelbase: Scalar,
) -> Float[JaxArray, "T 4 2 K"]:
    v = states[:, 3, :]
    delta = controls[:, 1, :]

    T, K = v.shape
    dt = time_step_size

    G = jnp.zeros((T, BICYCLE_D_X, BICYCLE_D_U, K))

    # ∂θ/∂δ = v / (L * cos²(δ)) * dt
    G = G.at[:, 2, 1, :].set(v / (wheelbase * jnp.cos(delta) ** 2) * dt)

    # ∂v/∂a = dt
    G = G.at[:, 3, 0, :].set(dt)

    return G


@jax.jit
@jaxtyped
def estimate_states(
    *,
    x_history: Float[JaxArray, "T K"],
    y_history: Float[JaxArray, "T K"],
    heading_history: Float[JaxArray, "T K"],
    time_step_size: Scalar,
    wheelbase: Scalar,
) -> EstimatedBicycleObstacleStates:
    return EstimatedBicycleObstacleStates(
        speed=(
            speed := estimate_speed(
                x_history=x_history,
                y_history=y_history,
                heading_history=heading_history,
                time_step_size=time_step_size,
            )
        ),
        accelerations=estimate_acceleration(
            x_history=x_history,
            y_history=y_history,
            heading_history=heading_history,
            speed_current=speed,
            time_step_size=time_step_size,
        ),
        steering_angles=estimate_steering_angle(
            heading_history=heading_history,
            speed_current=speed,
            time_step_size=time_step_size,
            wheelbase=wheelbase,
        ),
    )


@jax.jit
@jaxtyped
def estimate_speed(
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

        speeds = (
            delta_x * jnp.cos(heading) + delta_y * jnp.sin(heading)
        ) / time_step_size

        return speeds

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
def estimate_acceleration(
    *,
    x_history: Float[JaxArray, "T K"],
    y_history: Float[JaxArray, "T K"],
    heading_history: Float[JaxArray, "T K"],
    speed_current: Float[JaxArray, "K"],
    time_step_size: Scalar,
) -> Float[JaxArray, "K"]:
    horizon = x_history.shape[0]
    obstacle_count = x_history.shape[1]

    def estimate(
        *, speed_current: Float[JaxArray, "K"], speed_previous: Float[JaxArray, "K"]
    ) -> Float[JaxArray, "K"]:
        return (speed_current - speed_previous) / time_step_size

    if horizon < 3:
        return jnp.zeros(obstacle_count)

    return estimate(
        speed_current=speed_current,
        speed_previous=estimate_speed(
            x_history=x_history[:-1],
            y_history=y_history[:-1],
            heading_history=heading_history[:-1],
            time_step_size=time_step_size,
        ),
    )


@jax.jit
@jaxtyped
def estimate_steering_angle(
    *,
    heading_history: Float[JaxArray, "T K"],
    speed_current: Float[JaxArray, "K"],
    time_step_size: Scalar,
    wheelbase: Scalar,
) -> Float[JaxArray, "K"]:
    horizon = heading_history.shape[0]
    obstacle_count = heading_history.shape[1]

    def estimate(
        *,
        speed_current: Float[JaxArray, "K"],
        heading_current: Float[JaxArray, "K"],
        heading_previous: Float[JaxArray, "K"],
    ) -> Float[JaxArray, "K"]:
        heading_velocity = (heading_current - heading_previous) / time_step_size

        steering_angles = jnp.where(
            jnp.abs(speed_current) > 1e-6,
            jnp.arctan(heading_velocity * wheelbase / speed_current),
            jnp.zeros_like(speed_current),
        )

        return steering_angles

    if horizon < 2:
        return jnp.zeros(obstacle_count)

    return estimate(
        speed_current=speed_current,
        heading_current=heading_history[-1],
        heading_previous=heading_history[-2],
    )
