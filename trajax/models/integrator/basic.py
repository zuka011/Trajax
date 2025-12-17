from typing import Final, Protocol
from dataclasses import dataclass

from trajax.states import (
    NumPySimpleState as SimpleState,
    NumPySimpleStateBatch as SimpleStateBatch,
)
from trajax.models.integrator.common import (
    IntegratorModel,
    IntegratorState,
    IntegratorStateBatch,
    IntegratorControlInputSequence,
    IntegratorControlInputBatch,
)

from numtypes import Array, Dims

import numpy as np


NO_LIMITS: Final = (float("-inf"), float("inf"))


class NumPyIntegratorState[D_x: int](IntegratorState[D_x], Protocol): ...


class NumPyIntegratorStateBatch[T: int, D_x: int, M: int](
    IntegratorStateBatch[T, D_x, M], Protocol
): ...


class NumPyIntegratorControlInputSequence[T: int, D_u: int](
    IntegratorControlInputSequence[T, D_u], Protocol
): ...


class NumPyIntegratorControlInputBatch[T: int, D_u: int, M: int](
    IntegratorControlInputBatch[T, D_u, M], Protocol
): ...


@dataclass(kw_only=True, frozen=True)
class NumPyIntegratorModel(
    IntegratorModel[
        NumPyIntegratorState,
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

    async def simulate[T: int, D_u: int, D_x: int, M: int](
        self,
        inputs: IntegratorControlInputBatch[T, D_u, M],
        initial_state: IntegratorState[D_x],
    ) -> SimpleStateBatch[T, D_x, M]:
        initial = np.asarray(initial_state)
        clipped_inputs = np.clip(inputs, *self.velocity_limits)

        return SimpleStateBatch(
            simulate_with_state_limits(
                inputs=clipped_inputs,
                initial_state=initial,
                time_step=self.time_step,
                state_limits=self.state_limits,
            )
            if self.has_state_limits
            else simulate_without_state_limits(
                inputs=clipped_inputs, initial_state=initial, time_step=self.time_step
            )
        )

    async def step[T: int, D_u: int, D_x: int](
        self, input: IntegratorControlInputSequence[T, D_u], state: IntegratorState[D_x]
    ) -> SimpleState[D_x]:
        controls = np.asarray(input)
        current_state = np.asarray(state)

        clipped_control = np.clip(controls[0], *self.velocity_limits)
        new_state = np.clip(
            current_state + clipped_control * self.time_step, *self.state_limits
        )

        return SimpleState(array=new_state)

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


def simulate_without_state_limits[T: int, D_u: int, D_x: int, M: int](
    *,
    inputs: Array[Dims[T, D_u, M]],
    initial_state: Array[Dims[D_x]],
    time_step: float,
) -> Array[Dims[T, D_u, M]]:
    initial_broadcasted = initial_state[:, np.newaxis]
    states = initial_broadcasted + np.cumsum(inputs * time_step, axis=0)
    return states
