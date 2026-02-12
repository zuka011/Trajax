from typing import Self, overload, cast, Sequence, Final, Any
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
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
    UnicycleD_v,
    UNICYCLE_D_V,
    UnicycleD_u,
    UNICYCLE_D_U,
    UnicycleD_o,
    UNICYCLE_D_O,
    DynamicalModel,
    ObstacleModel,
    EstimatedObstacleStates,
)

from numtypes import Array, Dims, D, shape_of, array

import numpy as np

NO_LIMITS: Final = (float("-inf"), float("inf"))

type StateArray = Array[Dims[UnicycleD_x]]
type ControlInputSequenceArray[T: int] = Array[Dims[T, UnicycleD_u]]
type StateBatchArray[T: int, M: int] = Array[Dims[T, UnicycleD_x, M]]
type ControlInputBatchArray[T: int, M: int] = Array[Dims[T, UnicycleD_u, M]]

type StatesAtTimeStep[M: int] = Array[Dims[UnicycleD_x, M]]
type ControlInputsAtTimeStep[M: int] = Array[Dims[UnicycleD_u, M]]


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
    array: Array[Dims[UnicycleD_o, K]]

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

    def heading(self) -> Array[Dims[K]]:
        return self.array[2, :]

    @property
    def dimension(self) -> UnicycleD_o:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]


@dataclass(frozen=True)
class NumPyUnicycleObstacleStateSequences[T: int, K: int]:
    array: Array[Dims[T, UnicycleD_o, K]]

    @staticmethod
    def create(
        *,
        x: Array[Dims[T, K]],
        y: Array[Dims[T, K]],
        heading: Array[Dims[T, K]],
    ) -> "NumPyUnicycleObstacleStateSequences[T, K]":
        T, K = x.shape
        array = np.stack([x, y, heading], axis=1)

        assert shape_of(array, matches=(T, UNICYCLE_D_O, K))

        return NumPyUnicycleObstacleStateSequences(array)

    def __array__(
        self, dtype: DataType | None = None
    ) -> Array[Dims[T, UnicycleD_o, K]]:
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
    def dimension(self) -> UnicycleD_o:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]


@dataclass(frozen=True)
class NumPyUnicycleObstacleVelocities[K: int]:
    linear_velocities: Array[Dims[K]]
    angular_velocities: Array[Dims[K]]

    def zeroed(self, *, angular_velocity: bool) -> "NumPyUnicycleObstacleVelocities[K]":
        """Returns a version of the velocities with the angular velocity zeroed out."""
        return NumPyUnicycleObstacleVelocities(
            linear_velocities=self.linear_velocities,
            angular_velocities=np.zeros_like(self.angular_velocities)
            if angular_velocity
            else self.angular_velocities,
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[UnicycleD_v, K]]:
        return self._array

    @property
    def dimension(self) -> UnicycleD_v:
        return UNICYCLE_D_V

    @property
    def count(self) -> K:
        return self.linear_velocities.shape[0]

    @cached_property
    def _array(self) -> Array[Dims[UnicycleD_v, K]]:
        return np.stack([self.linear_velocities, self.angular_velocities], axis=0)


@dataclass(frozen=True)
class NumPyUnicycleObstacleControlInputSequences[T: int, K: int]:
    array: Array[Dims[T, UnicycleD_u, K]]

    @staticmethod
    def wrap[T_: int, K_: int](
        array: Array[Dims[T_, UnicycleD_u, K_]],
    ) -> "NumPyUnicycleObstacleControlInputSequences[T_, K_]":
        return NumPyUnicycleObstacleControlInputSequences(array)

    @staticmethod
    def create(
        *,
        linear_velocities: Array[Dims[T, K]],
        angular_velocities: Array[Dims[T, K]],
    ) -> "NumPyUnicycleObstacleControlInputSequences[T, K]":
        T, K = linear_velocities.shape
        array = np.stack([linear_velocities, angular_velocities], axis=1)

        assert shape_of(array, matches=(T, UNICYCLE_D_U, K))

        return NumPyUnicycleObstacleControlInputSequences(array)

    def __array__(
        self, dtype: DataType | None = None
    ) -> Array[Dims[T, UnicycleD_u, K]]:
        return self.array

    def linear_velocities(self) -> Array[Dims[T, K]]:
        return self.array[:, 0, :]

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> UnicycleD_u:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]


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


@dataclass(kw_only=True, frozen=True)
class NumPyUnicycleObstacleModel(
    ObstacleModel[
        NumPyUnicycleObstacleStatesHistory,
        NumPyUnicycleObstacleStates,
        NumPyUnicycleObstacleVelocities,
        NumPyUnicycleObstacleControlInputSequences,
        NumPyUnicycleObstacleStateSequences,
    ]
):
    """Estimates obstacle velocities from history and propagates unicycle kinematics forward."""

    time_step_size: float

    @staticmethod
    def create(*, time_step_size: float) -> "NumPyUnicycleObstacleModel":
        """Creates a NumPy unicycle obstacle model."""
        return NumPyUnicycleObstacleModel(time_step_size=time_step_size)

    def estimate_state_from[K: int](
        self, history: NumPyUnicycleObstacleStatesHistory[int, K]
    ) -> EstimatedObstacleStates[
        NumPyUnicycleObstacleStates[K], NumPyUnicycleObstacleVelocities[K]
    ]:
        assert history.horizon > 0, "History must have at least one time step."

        if history.horizon < 2:
            linear_velocities = np.zeros((history.count))
            angular_velocities = np.zeros((history.count))
        else:
            linear_velocities = estimate_speeds_from(
                history, time_step_size=self.time_step_size
            )
            angular_velocities = estimate_angular_velocities_from(
                history, time_step_size=self.time_step_size
            )

        assert shape_of(linear_velocities, matches=(history.count,))
        assert shape_of(angular_velocities, matches=(history.count,))

        return EstimatedObstacleStates(
            states=NumPyUnicycleObstacleStates.create(
                x=history.x()[-1],
                y=history.y()[-1],
                heading=history.heading()[-1],
            ),
            velocities=NumPyUnicycleObstacleVelocities(
                linear_velocities=linear_velocities,
                angular_velocities=angular_velocities,
            ),
        )

    def input_to_maintain[K: int](
        self,
        velocities: NumPyUnicycleObstacleVelocities[K],
        *,
        states: NumPyUnicycleObstacleStates[K],
        horizon: int,
    ) -> NumPyUnicycleObstacleControlInputSequences[int, K]:
        return NumPyUnicycleObstacleControlInputSequences.create(
            linear_velocities=np.tile(
                velocities.linear_velocities[np.newaxis, :], (horizon, 1)
            ),
            angular_velocities=np.tile(
                velocities.angular_velocities[np.newaxis, :], (horizon, 1)
            ),
        )

    def forward[T: int, K: int](
        self,
        *,
        current: NumPyUnicycleObstacleStates[K],
        inputs: NumPyUnicycleObstacleControlInputSequences[T, K],
    ) -> NumPyUnicycleObstacleStateSequences[T, K]:
        result = simulate(
            inputs.array,
            current.array,
            time_step_size=self.time_step_size,
            speed_limits=(float("-inf"), float("inf")),
            angular_velocity_limits=(float("-inf"), float("inf")),
        )

        return NumPyUnicycleObstacleStateSequences.create(
            x=result[:, 0, :],
            y=result[:, 1, :],
            heading=result[:, 2, :],
        )

    def state_jacobian[T: int, K: int](
        self,
        *,
        states: NumPyUnicycleObstacleStateSequences[T, K],
        inputs: NumPyUnicycleObstacleControlInputSequences[T, K],
    ) -> Array[Dims[T, UnicycleD_o, UnicycleD_o, K]]:
        return state_jacobian(
            heading=states.heading(),
            speed=inputs.linear_velocities(),
            time_step_size=self.time_step_size,
        )

    def input_jacobian[T: int, K: int](
        self,
        *,
        states: NumPyUnicycleObstacleStateSequences[T, K],
        inputs: NumPyUnicycleObstacleControlInputSequences[T, K],
    ) -> Array[Dims[T, UnicycleD_o, UnicycleD_u, K]]:
        return input_jacobian(
            heading=states.heading(),
            time_step_size=self.time_step_size,
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


def state_jacobian[T: int, K: int](
    heading: Array[Dims[T, K]],
    speed: Array[Dims[T, K]],
    *,
    time_step_size: float,
) -> Array[Dims[T, UnicycleD_o, UnicycleD_o, K]]:
    """Computes the state Jacobian F = ∂f/∂x for the unicycle model."""
    v, theta = speed, heading

    T, K = heading.shape
    F = np.zeros((T, UNICYCLE_D_O, UNICYCLE_D_O, K))

    dt = time_step_size

    F[:, 0, 0, :] = 1.0
    F[:, 1, 1, :] = 1.0
    F[:, 2, 2, :] = 1.0

    F[:, 0, 2, :] = -v * np.sin(theta) * dt
    F[:, 1, 2, :] = v * np.cos(theta) * dt

    assert shape_of(
        F, matches=(T, UNICYCLE_D_O, UNICYCLE_D_O, K), name="state_jacobian"
    )

    return F


def input_jacobian[T: int, K: int](
    heading: Array[Dims[T, K]],
    *,
    time_step_size: float,
) -> Array[Dims[T, UnicycleD_o, UnicycleD_u, K]]:
    """Computes the input Jacobian G = ∂f/∂u for the unicycle model.

    For unicycle model: u = [linear_velocity, angular_velocity]
    G describes how control input uncertainty enters the state dynamics.
    """
    theta = heading

    T, K = heading.shape
    G = np.zeros((T, UNICYCLE_D_O, UNICYCLE_D_U, K))

    dt = time_step_size

    # ∂x/∂v = cos(θ) * dt
    G[:, 0, 0, :] = np.cos(theta) * dt

    # ∂y/∂v = sin(θ) * dt
    G[:, 1, 0, :] = np.sin(theta) * dt

    # ∂θ/∂ω = dt
    G[:, 2, 1, :] = dt

    assert shape_of(
        G, matches=(T, UNICYCLE_D_O, UNICYCLE_D_U, K), name="input_jacobian"
    )

    return G


def estimate_speeds_from[K: int](
    history: NumPyUnicycleObstacleStatesHistory[int, K], *, time_step_size: float
) -> Array[Dims[K]]:
    assert history.horizon >= 2, (
        "At least two history steps are required to estimate speed."
    )

    delta_x = history.x()[-1] - history.x()[-2]
    delta_y = history.y()[-1] - history.y()[-2]

    speeds = np.sqrt(delta_x**2 + delta_y**2) / time_step_size

    assert shape_of(speeds, matches=(history.count,), name="estimated speeds")

    return speeds


def estimate_angular_velocities_from[K: int](
    history: NumPyUnicycleObstacleStatesHistory[int, K], *, time_step_size: float
) -> Array[Dims[K]]:
    assert history.horizon >= 2, (
        "At least two history steps are required to estimate angular velocity."
    )

    angular_velocities = (
        history.heading()[-1] - history.heading()[-2]
    ) / time_step_size

    assert shape_of(
        angular_velocities,
        matches=(history.count,),
        name="estimated angular velocities",
    )

    return angular_velocities
