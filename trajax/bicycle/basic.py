from dataclasses import dataclass

from trajax.type import DataType
from trajax.bicycle.common import D_x, D_X, D_u

from numtypes import Array, Dims, D, shape_of
import numpy as np


type StateArray = Array[Dims[D_x]]
type ControlInputArray = Array[Dims[D_u]]
type ControlInputSequenceArray[T: int] = Array[Dims[T, D_u]]

type StateBatchArray[T: int, M: int] = Array[Dims[T, D_x, M]]
type ControlInputBatchArray[T: int, M: int] = Array[Dims[T, D_u, M]]


@dataclass(frozen=True)
class NumPyState:
    state: StateArray

    def __array__(self, dtype: DataType | None = None) -> StateArray:
        return self.state

    @property
    def x(self) -> float:
        return self.state[0]

    @property
    def y(self) -> float:
        return self.state[1]

    @property
    def theta(self) -> float:
        return self.state[2]

    @property
    def v(self) -> float:
        return self.state[3]


@dataclass(frozen=True)
class NumPyStateBatch[T: int, M: int]:
    states: StateBatchArray[T, M]

    def __array__(self, dtype: DataType | None = None) -> StateBatchArray[T, M]:
        return self.states

    def orientations(self) -> Array[Dims[T, M]]:
        return self.states[:, 2, :]

    def velocities(self) -> Array[Dims[T, M]]:
        return self.states[:, 3, :]

    @property
    def positions(self) -> "NumPyPositions":
        return NumPyPositions(state=self)


@dataclass(frozen=True)
class NumPyPositions[T: int, M: int]:
    state: NumPyStateBatch[T, M]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[2], M]]:
        return self.state.states[:, :2, :]

    def x(self) -> Array[Dims[T, M]]:
        return self.state.states[:, 0, :]

    def y(self) -> Array[Dims[T, M]]:
        return self.state.states[:, 1, :]


@dataclass(frozen=True)
class NumPyControlInputSequence[T: int]:
    inputs: ControlInputSequenceArray[T]

    def __array__(self, dtype: DataType | None = None) -> ControlInputSequenceArray[T]:
        return self.inputs

    # TODO: Add methods: `similar` and `dimension`


@dataclass(frozen=True)
class NumPyControlInputBatch[T: int, M: int]:
    inputs: ControlInputBatchArray[T, M]

    def __array__(self, dtype: DataType | None = None) -> ControlInputBatchArray[T, M]:
        return self.inputs

    @property
    def rollout_count(self) -> M:
        return self.inputs.shape[2]

    @property
    def horizon(self) -> T:
        return self.inputs.shape[0]


@dataclass(kw_only=True, frozen=True)
class NumPyBicycleModel:
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

    async def simulate[T: int, M: int](
        self, inputs: NumPyControlInputBatch[T, M], initial_state: NumPyState
    ) -> NumPyStateBatch[T, M]:
        horizon = inputs.horizon
        rollout_count = inputs.rollout_count
        states = np.zeros((horizon, 4, rollout_count))

        x = np.full(rollout_count, initial_state.x)
        y = np.full(rollout_count, initial_state.y)
        theta = np.full(rollout_count, initial_state.theta)
        v = np.full(rollout_count, initial_state.v)

        for t in range(horizon):
            acceleration = np.clip(
                inputs.inputs[t, 0, :], self.min_acceleration, self.max_acceleration
            )
            steering = np.clip(
                inputs.inputs[t, 1, :], self.min_steering, self.max_steering
            )

            x = x + v * np.cos(theta) * self.time_step_size
            y = y + v * np.sin(theta) * self.time_step_size
            theta = theta + v * np.tan(steering) / self.wheelbase * self.time_step_size
            v = np.clip(
                v + acceleration * self.time_step_size, self.min_speed, self.max_speed
            )

            states[t, 0, :] = x
            states[t, 1, :] = y
            states[t, 2, :] = theta
            states[t, 3, :] = v

        assert shape_of(
            states, matches=(horizon, D_X, rollout_count), name="simulated states"
        )

        return NumPyStateBatch(states)

    async def step[T: int](
        self, input: NumPyControlInputSequence[T], state: NumPyState
    ) -> NumPyState:
        raise NotImplementedError(
            "Single step simulation is not implemented for NumPyBicycleModel."
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
