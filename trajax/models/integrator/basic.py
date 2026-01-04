from typing import Final
from dataclasses import dataclass

from trajax.types import (
    DynamicalModel,
    ObstacleModel,
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
    array: Array[Dims[D_o, K]]


@dataclass(frozen=True)
class NumPyIntegratorObstacleStateSequences[T: int, D_o: int, K: int]:
    array: Array[Dims[T, D_o, K]]


@dataclass(frozen=True)
class NumPyIntegratorObstacleVelocities[D_o: int, K: int]:
    array: Array[Dims[D_o, K]]


@dataclass(frozen=True)
class NumPyIntegratorObstacleControlInputSequences[T: int, D_o: int, K: int]:
    array: Array[Dims[T, D_o, K]]


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
    _time_step_size: float
    state_limits: tuple[float, float]
    velocity_limits: tuple[float, float]

    @staticmethod
    def create(
        *,
        time_step_size: float,
        state_limits: tuple[float, float] | None = None,
        velocity_limits: tuple[float, float] | None = None,
    ) -> "NumPyIntegratorModel":
        """A NumPy integrator model where state = cumulative sum of controls.

        This model represents a particle that moves according to velocity commands.
        State dimension must equal control dimension (D_x == D_u).

        x_{t+1} = clip(x_t + clip(u_t, velocity_limits) * dt, state_limits)

        Args:
            time_step_size: The time step size for the integrator.
            state_limits: Optional tuple of (min, max) limits for the state values.
            velocity_limits: Optional tuple of (min, max) limits for the velocity inputs.
        """
        return NumPyIntegratorModel(
            _time_step_size=time_step_size,
            state_limits=state_limits if state_limits is not None else NO_LIMITS,
            velocity_limits=velocity_limits
            if velocity_limits is not None
            else NO_LIMITS,
        )

    def simulate[T: int, D_x: int, M: int](
        self,
        inputs: NumPyIntegratorControlInputBatch[T, D_x, M],
        initial_state: NumPyIntegratorState[D_x],
    ) -> SimpleStateBatch[T, D_x, M]:
        clipped_inputs = np.clip(inputs.array, *self.velocity_limits)

        return SimpleStateBatch(
            simulate_with_state_limits(
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
        new_state = np.clip(
            state.array + clipped_control * self.time_step_size, *self.state_limits
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
        NumPyIntegratorObstacleVelocities,
        NumPyIntegratorObstacleControlInputSequences,
        NumPyIntegratorObstacleStateSequences,
    ]
):
    time_step: float

    @staticmethod
    def create(*, time_step_size: float) -> "NumPyIntegratorObstacleModel":
        """Creates a NumPy integrator obstacle model.

        See `NumPyIntegratorModel.create` for details on the integrator dynamics.
        """
        return NumPyIntegratorObstacleModel(time_step=time_step_size)

    def estimate_state_from[D_o: int, K: int](
        self, history: NumPyIntegratorObstacleStatesHistory[int, D_o, K]
    ) -> EstimatedObstacleStates[
        NumPyIntegratorObstacleStates[D_o, K], NumPyIntegratorObstacleVelocities[D_o, K]
    ]:
        assert history.horizon > 0, "History must have at least one time step."

        if history.horizon < 2:
            velocities = np.zeros((history.dimension, history.count))
        else:
            velocities = (
                history.array[-1, :, :] - history.array[-2, :, :]
            ) / self.time_step

        assert shape_of(velocities, matches=(history.dimension, history.count))

        return EstimatedObstacleStates(
            states=NumPyIntegratorObstacleStates(history.array[-1, :, :]),
            velocities=NumPyIntegratorObstacleVelocities(velocities),
        )

    def input_to_maintain[T: int, D_o: int, K: int](
        self,
        velocities: NumPyIntegratorObstacleVelocities[D_o, K],
        *,
        states: NumPyIntegratorObstacleStates[D_o, K],
        horizon: T,
    ) -> NumPyIntegratorObstacleControlInputSequences[T, D_o, K]:
        return NumPyIntegratorObstacleControlInputSequences(
            np.tile(velocities.array[np.newaxis, :, :], (horizon, 1, 1))
        )

    def forward[T: int, D_o: int, K: int](
        self,
        *,
        current: NumPyIntegratorObstacleStates[D_o, K],
        inputs: NumPyIntegratorObstacleControlInputSequences[T, D_o, K],
    ) -> NumPyIntegratorObstacleStateSequences[T, D_o, K]:
        result = simulate(
            inputs=inputs.array, initial_states=current.array, time_step=self.time_step
        )

        return NumPyIntegratorObstacleStateSequences(result)


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


def simulate[T: int, D_x: int, N: int](
    *,
    inputs: Array[Dims[T, D_x, N]],
    initial_states: Array[Dims[D_x, N]],
    time_step: float,
) -> Array[Dims[T, D_x, N]]:
    states = initial_states + np.cumsum(inputs * time_step, axis=0)
    return states
