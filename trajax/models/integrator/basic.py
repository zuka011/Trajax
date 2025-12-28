from typing import Final
from dataclasses import dataclass

from trajax.types import (
    DynamicalModel,
    ObstacleModel,
    NumPyIntegratorState,
    NumPyIntegratorStateBatch,
    NumPyIntegratorControlInputSequence,
    NumPyIntegratorControlInputBatch,
    NumPyObstacleStatesHistory,
    EstimatedObstacleStates,
    D_o,
)
from trajax.states import (
    NumPySimpleState as SimpleState,
    NumPySimpleStateBatch as SimpleStateBatch,
)
from trajax.obstacles import NumPyObstacleStates

from numtypes import Array, Dims, shape_of

import numpy as np


NO_LIMITS: Final = (float("-inf"), float("inf"))


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorObstacleStates[K: int]:
    x: Array[Dims[D_o, K]]
    y: Array[Dims[D_o, K]]
    heading: Array[Dims[D_o, K]]

    @property
    def array(self) -> Array[Dims[D_o, K]]:
        return np.stack([self.x, self.y, self.heading], axis=0)


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorObstacleVelocities[K: int]:
    x: Array[Dims[K]]
    y: Array[Dims[K]]
    heading: Array[Dims[K]]

    @property
    def array(self) -> Array[Dims[D_o, K]]:
        return np.stack([self.x, self.y, self.heading], axis=0)


@dataclass(frozen=True)
class NumPyIntegratorObstacleControlInputSequences[T: int, K: int]:
    array: Array[Dims[T, D_o, K]]


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorModel(
    DynamicalModel[
        NumPyIntegratorState,
        NumPyIntegratorStateBatch,
        NumPyIntegratorControlInputSequence,
        NumPyIntegratorControlInputBatch,
    ]
):
    time_step: float
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
            time_step=time_step_size,
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
                time_step=self.time_step,
                state_limits=self.state_limits,
            )
            if self.has_state_limits
            else simulate(
                inputs=clipped_inputs,
                initial_states=initial_state.array[:, np.newaxis],
                time_step=self.time_step,
            )
        )

    def step[T: int, D_x: int](
        self,
        input: NumPyIntegratorControlInputSequence[T, D_x],
        state: NumPyIntegratorState[D_x],
    ) -> SimpleState[D_x]:
        clipped_control = np.clip(input.array[0], *self.velocity_limits)
        new_state = np.clip(
            state.array + clipped_control * self.time_step, *self.state_limits
        )

        return SimpleState(new_state)

    @property
    def has_state_limits(self) -> bool:
        return self.state_limits != NO_LIMITS

    @property
    def min_state(self) -> float:
        return self.state_limits[0]

    @property
    def max_state(self) -> float:
        return self.state_limits[1]

    @property
    def min_velocity(self) -> float:
        return self.velocity_limits[0]

    @property
    def max_velocity(self) -> float:
        return self.velocity_limits[1]


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorObstacleModel(
    ObstacleModel[
        NumPyObstacleStatesHistory,
        NumPyIntegratorObstacleStates,
        NumPyIntegratorObstacleVelocities,
        NumPyIntegratorObstacleControlInputSequences,
        NumPyObstacleStates,
    ]
):
    time_step: float

    @staticmethod
    def create(*, time_step_size: float) -> "NumPyIntegratorObstacleModel":
        """Creates a NumPy integrator obstacle model.

        See `NumPyIntegratorModel.create` for details on the integrator dynamics.
        """
        return NumPyIntegratorObstacleModel(time_step=time_step_size)

    def estimate_state_from[K: int](
        self, history: NumPyObstacleStatesHistory[int, K]
    ) -> EstimatedObstacleStates[
        NumPyIntegratorObstacleStates[K], NumPyIntegratorObstacleVelocities[K]
    ]:
        x = history.x()
        y = history.y()
        heading = history.heading()

        if history.horizon < 2:
            v_x = np.zeros((history.count,))
            v_y = np.zeros((history.count,))
            v_heading = np.zeros((history.count,))
        else:
            v_x = (x[-1, :] - x[-2, :]) / self.time_step
            v_y = (y[-1, :] - y[-2, :]) / self.time_step
            v_heading = (heading[-1, :] - heading[-2, :]) / self.time_step

        assert shape_of(v_x, matches=(history.count,))
        assert shape_of(v_y, matches=(history.count,))
        assert shape_of(v_heading, matches=(history.count,))

        return EstimatedObstacleStates(
            states=NumPyIntegratorObstacleStates(
                x=x[-1, :], y=y[-1, :], heading=heading[-1, :]
            ),
            velocities=NumPyIntegratorObstacleVelocities(
                x=v_x, y=v_y, heading=v_heading
            ),
        )

    def input_to_maintain[T: int, K: int](
        self,
        velocities: NumPyIntegratorObstacleVelocities[K],
        *,
        states: NumPyIntegratorObstacleStates[K],
        horizon: T,
    ) -> NumPyIntegratorObstacleControlInputSequences[T, K]:
        return NumPyIntegratorObstacleControlInputSequences(
            np.tile(velocities.array[np.newaxis, :, :], (horizon, 1, 1))
        )

    def forward[T: int, K: int](
        self,
        *,
        current: NumPyIntegratorObstacleStates[K],
        input: NumPyIntegratorObstacleControlInputSequences[T, K],
    ) -> NumPyObstacleStates[T, K]:
        result = simulate(
            inputs=input.array,
            initial_states=current.array,
            time_step=self.time_step,
        )

        return NumPyObstacleStates.create(
            x=result[:, 0, :], y=result[:, 1, :], heading=result[:, 2, :]
        )


def simulate_with_state_limits[T: int, D_u: int, D_x: int, M: int](
    *,
    inputs: Array[Dims[T, D_u, M]],
    initial_state: Array[Dims[D_x]],
    time_step: float,
    state_limits: tuple[float, float],
) -> Array[Dims[T, D_u, M]]:
    deltas = inputs * time_step
    initial_broadcasted = initial_state[:, np.newaxis, np.newaxis]

    states = np.empty_like(deltas)
    states[0] = np.clip(initial_broadcasted + deltas[0], *state_limits)

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
