from typing import Final, cast
from dataclasses import dataclass

from trajax.types import (
    DynamicalModel,
    ObstacleModel,
    ObstacleStateEstimator,
    NumPyIntegratorState,
    NumPyIntegratorStateSequence,
    NumPyIntegratorStateBatch,
    NumPyIntegratorControlInputSequence,
    NumPyIntegratorControlInputBatch,
    NumPyIntegratorObstacleStatesHistory,
    EstimatedObstacleStates,
)
from trajax.states import (
    NumPySimpleState as SimpleState,
    NumPySimpleStateSequence as SimpleStateSequence,
    NumPySimpleStateBatch as SimpleStateBatch,
    NumPySimpleControlInputBatch as SimpleControlInputBatch,
)

from numtypes import Array, Dims, shape_of

import numpy as np


NO_LIMITS: Final = (float("-inf"), float("inf"))


@dataclass(frozen=True)
class NumPyIntegratorObstacleStates[D_o: int, K: int]:
    """Obstacle states represented in integrator model coordinates."""

    _array: Array[Dims[D_o, K]]

    def __array__(self, dtype: None | type = None) -> Array[Dims[D_o, K]]:
        return self.array

    @property
    def dimension(self) -> D_o:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[D_o, K]]:
        return self._array


@dataclass(frozen=True)
class NumPyIntegratorObstacleStateSequences[T: int, D_o: int, K: int]:
    """Time-indexed obstacle state sequences for integrator model obstacles."""

    array: Array[Dims[T, D_o, K]]

    def __array__(self, dtype: None | type = None) -> Array[Dims[T, D_o, K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_o:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]


@dataclass(frozen=True)
class NumPyIntegratorObstacleInputs[D_o: int, K: int]:
    array: Array[Dims[D_o, K]]

    def __array__(self, dtype: None | type = None) -> Array[Dims[D_o, K]]:
        return self.array

    def zeroed(self, *, at: tuple[int, ...]) -> "NumPyIntegratorObstacleInputs[D_o, K]":
        """Returns new obstacle inputs with inputs at specified state dimensions zeroed out."""

        zeroed_array = self.array.copy()
        zeroed_array[at, :] = 0.0

        return NumPyIntegratorObstacleInputs(zeroed_array)

    @property
    def dimension(self) -> D_o:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]


@dataclass(frozen=True)
class NumPyIntegratorObstacleControlInputSequences[T: int, D_o: int, K: int]:
    array: Array[Dims[T, D_o, K]]

    def __array__(self, dtype: None | type = None) -> Array[Dims[T, D_o, K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_o:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorModel(
    DynamicalModel[
        NumPyIntegratorState,
        NumPyIntegratorStateSequence,
        NumPyIntegratorStateBatch,
        NumPyIntegratorControlInputSequence,
        NumPyIntegratorControlInputBatch,
    ]
):
    """Point-mass model with direct position control, used for obstacle prediction."""

    _time_step_size: float
    state_limits: tuple[float, float]
    velocity_limits: tuple[float, float]
    periodic: bool

    @staticmethod
    def create(
        *,
        time_step_size: float,
        state_limits: tuple[float, float] | None = None,
        velocity_limits: tuple[float, float] | None = None,
        periodic: bool = False,
    ) -> "NumPyIntegratorModel":
        """A NumPy integrator model where state = cumulative sum of controls.

        This model represents a particle that moves according to velocity commands.
        State dimension must equal control dimension (D_x == D_u).

        $$x_{t+1} = \\text{clip}(x_t + \\text{clip}(u_t,\\; v_{\\text{lim}}) \\cdot \\Delta t,\\; s_{\\text{lim}})$$

        Args:
            time_step_size: The time step size for the integrator.
            state_limits: Optional tuple of (min, max) limits for the state values.
            velocity_limits: Optional tuple of (min, max) limits for the velocity inputs.
            periodic: Whether to apply periodic boundary conditions based on state_limits.
        """
        return NumPyIntegratorModel(
            _time_step_size=time_step_size,
            state_limits=state_limits if state_limits is not None else NO_LIMITS,
            velocity_limits=velocity_limits
            if velocity_limits is not None
            else NO_LIMITS,
            periodic=periodic,
        )

    def __post_init__(self) -> None:
        if self.periodic:
            validate_periodic_state_limits(self.state_limits)

    def simulate[T: int, D_x: int, M: int](
        self,
        inputs: NumPyIntegratorControlInputBatch[T, D_x, M],
        initial_state: NumPyIntegratorState[D_x],
    ) -> SimpleStateBatch[T, D_x, M]:
        clipped_inputs = np.clip(inputs.array, *self.velocity_limits)

        return SimpleStateBatch(
            wrap_periodic_batch(
                states=simulate(
                    inputs=clipped_inputs,
                    initial_states=initial_state.array[:, np.newaxis],
                    time_step=self.time_step_size,
                ),
                state_limits=self.state_limits,
            )
            if self.periodic
            else simulate_with_state_limits(
                inputs=clipped_inputs,
                initial_state=initial_state.array,
                time_step=self.time_step_size,
                state_limits=self.state_limits,
            )
            if self.has_state_limits
            else simulate(
                inputs=clipped_inputs,
                initial_states=initial_state.array[:, np.newaxis],
                time_step=self.time_step_size,
            )
        )

    def step[T: int, D_x: int](
        self,
        inputs: NumPyIntegratorControlInputSequence[T, D_x],
        state: NumPyIntegratorState[D_x],
    ) -> SimpleState[D_x]:
        clipped_control = np.clip(inputs.array[0], *self.velocity_limits)
        unbounded = state.array + clipped_control * self.time_step_size
        new_state = (
            wrap_periodic_state(state=unbounded, state_limits=self.state_limits)
            if self.periodic
            else np.clip(unbounded, *self.state_limits)
        )

        return SimpleState(new_state)

    def forward[T: int, D_x: int](
        self,
        inputs: NumPyIntegratorControlInputSequence[T, D_x],
        state: NumPyIntegratorState[D_x],
    ) -> SimpleStateSequence[T, D_x]:
        return self.simulate(
            inputs=SimpleControlInputBatch.of(inputs), initial_state=state
        ).rollout(0)

    @property
    def time_step_size(self) -> float:
        return self._time_step_size

    @property
    def has_state_limits(self) -> bool:
        return self.state_limits != NO_LIMITS


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorObstacleModel(
    ObstacleModel[
        NumPyIntegratorObstacleStatesHistory,
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleInputs,
        NumPyIntegratorObstacleStateSequences,
    ]
):
    """Propagates integrator dynamics forward with constant velocity."""

    time_step: float

    @staticmethod
    def create(*, time_step_size: float) -> "NumPyIntegratorObstacleModel":
        """Creates a NumPy integrator obstacle model.

        See `NumPyIntegratorModel.create` for details on the integrator dynamics.
        """
        return NumPyIntegratorObstacleModel(time_step=time_step_size)

    def forward[T: int, D_o: int, K: int](
        self,
        *,
        current: NumPyIntegratorObstacleStates[D_o, K],
        inputs: NumPyIntegratorObstacleInputs[D_o, K],
        horizon: T,
    ) -> NumPyIntegratorObstacleStateSequences[T, D_o, K]:
        input_sequences = self._input_to_maintain(inputs, horizon=horizon)

        result = simulate(
            inputs=input_sequences.array,
            initial_states=current.array,
            time_step=self.time_step,
        )

        return NumPyIntegratorObstacleStateSequences(result)

    def state_jacobian[T: int, D_o: int, K: int](
        self,
        *,
        states: NumPyIntegratorObstacleStateSequences[T, D_o, K],
        inputs: NumPyIntegratorObstacleInputs[D_o, K],
    ) -> Array[Dims[T, D_o, D_o, K]]:
        raise NotImplementedError(
            "State Jacobian is not implemented for NumPyIntegratorObstacleModel."
        )

    def input_jacobian[T: int, D_o: int, K: int](
        self,
        *,
        states: NumPyIntegratorObstacleStateSequences[T, D_o, K],
        inputs: NumPyIntegratorObstacleInputs[D_o, K],
    ) -> Array[Dims[T, D_o, D_o, K]]:
        raise NotImplementedError(
            "Input Jacobian is not implemented for NumPyIntegratorObstacleModel."
        )

    def _input_to_maintain[T: int, D_o: int, K: int](
        self,
        inputs: NumPyIntegratorObstacleInputs[D_o, K],
        *,
        horizon: T,
    ) -> NumPyIntegratorObstacleControlInputSequences[T, D_o, K]:
        return NumPyIntegratorObstacleControlInputSequences(
            np.tile(inputs.array[np.newaxis, :, :], (horizon, 1, 1))
        )


@dataclass(frozen=True)
class NumPyFiniteDifferenceIntegratorStateEstimator(
    ObstacleStateEstimator[
        NumPyIntegratorObstacleStatesHistory,
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleInputs,
    ]
):
    time_step_size: float

    @staticmethod
    def create(
        *, time_step_size: float
    ) -> "NumPyFiniteDifferenceIntegratorStateEstimator":
        return NumPyFiniteDifferenceIntegratorStateEstimator(
            time_step_size=time_step_size
        )

    def estimate_from[D_o: int, K: int, T: int = int](
        self, history: NumPyIntegratorObstacleStatesHistory[T, D_o, K]
    ) -> EstimatedObstacleStates[
        NumPyIntegratorObstacleStates[D_o, K], NumPyIntegratorObstacleInputs[D_o, K]
    ]:
        velocities = self.estimate_velocities_from(history)

        return EstimatedObstacleStates(
            states=NumPyIntegratorObstacleStates(history.array[-1, :, :]),
            inputs=NumPyIntegratorObstacleInputs(velocities),
        )

    def estimate_velocities_from[D_o: int, K: int, T: int = int](
        self, history: NumPyIntegratorObstacleStatesHistory[T, D_o, K]
    ) -> Array[Dims[D_o, K]]:
        """Estimates velocities from position history using finite differences.

        $$v_t = (x_t - x_{t-1}) / (\\Delta t)$$
        """
        if history.horizon < 2:
            return cast(
                Array[Dims[D_o, K]], np.zeros((history.dimension, history.count))
            )

        return self._estimate_velocities_from(
            current=history.array[-1, :, :],
            previous=history.array[-2, :, :],
        )

    def _estimate_velocities_from[D_o: int, K: int](
        self,
        *,
        current: Array[Dims[D_o, K]],
        previous: Array[Dims[D_o, K]],
    ) -> Array[Dims[D_o, K]]:
        velocities = (current - previous) / self.time_step_size

        assert shape_of(
            velocities, matches=(current.shape[0], current.shape[1]), name="velocities"
        )

        return velocities


def validate_periodic_state_limits(state_limits: tuple[float, float] | None) -> None:
    assert state_limits is not None, (
        "Periodic boundaries require explicit state limits."
    )

    lower, upper = state_limits
    assert np.isfinite(lower) and np.isfinite(upper), (
        "Periodic boundaries must be finite."
    )

    assert upper > lower, (
        "Periodic boundaries require upper limit to be greater than lower limit."
    )


def simulate_with_state_limits[T: int, D_x: int, M: int](
    *,
    inputs: Array[Dims[T, D_x, M]],
    initial_state: Array[Dims[D_x]],
    time_step: float,
    state_limits: tuple[float, float],
) -> Array[Dims[T, D_x, M]]:
    deltas = inputs * time_step

    states = np.empty_like(deltas)
    states[0] = np.clip(initial_state[:, np.newaxis] + deltas[0], *state_limits)

    for t in range(1, deltas.shape[0]):
        states[t] = np.clip(states[t - 1] + deltas[t], *state_limits)

    return states


def wrap_periodic_state[D_x: int](
    *,
    state: Array[Dims[D_x]],
    state_limits: tuple[float, float],
) -> Array[Dims[D_x]]:
    lower, upper = state_limits
    period = upper - lower
    wrapped = lower + np.mod(state - lower, period)
    return np.where(state == upper, upper, wrapped)


def wrap_periodic_batch[T: int, D_x: int, M: int](
    *,
    states: Array[Dims[T, D_x, M]],
    state_limits: tuple[float, float],
) -> Array[Dims[T, D_x, M]]:
    lower, upper = state_limits
    period = upper - lower
    wrapped = lower + np.mod(states - lower, period)
    return np.where(states == upper, upper, wrapped)


def simulate[T: int, D_x: int, N: int](
    *,
    inputs: Array[Dims[T, D_x, N]],
    initial_states: Array[Dims[D_x, N]],
    time_step: float,
) -> Array[Dims[T, D_x, N]]:
    states = initial_states + np.cumsum(inputs * time_step, axis=0)
    return states
