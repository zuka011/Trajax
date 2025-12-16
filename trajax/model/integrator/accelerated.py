from typing import Protocol, Final
from dataclasses import dataclass

from trajax.type import jaxtyped
from trajax.states.simple.accelerated import (
    State as SimpleState,
    StateBatch as SimpleStateBatch,
)
from trajax.model.integrator.common import (
    IntegratorModel,
    State as AnyState,
    ControlInputSequence as AnyControlInputSequence,
    ControlInputBatch as AnyControlInputBatch,
)

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


NO_LIMITS: Final = (float("-inf"), float("inf"))


class State[D_x: int = int](AnyState[D_x], Protocol):
    @property
    def array(self) -> Float[JaxArray, "D_x"]:
        """Returns the underlying JAX array."""
        ...


class ControlInputSequence[T: int = int, D_u: int = int](
    AnyControlInputSequence[T, D_u], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        """Returns the underlying JAX array."""
        ...

    @property
    def dimension(self) -> D_u:
        """Returns the dimension of the control input."""
        ...


class ControlInputBatch[T: int = int, D_u: int = int, M: int = int](
    AnyControlInputBatch[T, D_u, M], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_u M"]:
        """Returns the underlying JAX array."""
        ...


@dataclass(kw_only=True, frozen=True)
class JaxIntegratorModel(
    IntegratorModel[
        State, SimpleState, SimpleStateBatch, ControlInputSequence, ControlInputBatch
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
    ) -> "JaxIntegratorModel":
        """A JAX integrator model where state = cumulative sum of controls.

        This model represents a particle that moves according to velocity commands.
        State dimension must equal control dimension (D_x == D_u).

        x_{t+1} = clip(x_t + clip(u_t, velocity_limits) * dt, state_limits)

        Args:
            time_step_size: The time step size for the integrator.
            state_limits: Optional tuple of (min, max) limits for the state values.
            velocity_limits: Optional tuple of (min, max) limits for the velocity inputs.
        """

        return JaxIntegratorModel(
            time_step=time_step_size,
            state_limits=state_limits if state_limits is not None else NO_LIMITS,
            velocity_limits=velocity_limits
            if velocity_limits is not None
            else NO_LIMITS,
        )

    async def simulate[T: int, D_u: int, D_x: int, M: int](
        self, inputs: ControlInputBatch[T, D_u, M], initial_state: State[D_x]
    ) -> SimpleStateBatch[T, D_x, M]:
        return SimpleStateBatch(
            integrator_simulate(
                controls=inputs.array,
                initial_state=initial_state.array,
                time_step=self.time_step,
                state_limits=self.state_limits,
                velocity_limits=self.velocity_limits,
                state_dimension=initial_state.dimension,
                rollout_count=inputs.rollout_count,
            )
        )

    async def step[T: int, D_u: int, D_x: int](
        self, input: ControlInputSequence[T, D_u], state: State[D_x]
    ) -> SimpleState[D_x]:
        return SimpleState(
            integrator_step(
                control=input.array,
                state=state.array,
                time_step=self.time_step,
                state_limits=self.state_limits,
                velocity_limits=self.velocity_limits,
            )
        )

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


@jax.jit(static_argnames=("state_dimension", "rollout_count"))
@jaxtyped
def integrator_simulate(
    *,
    controls: Float[JaxArray, "T D_u M"],
    initial_state: Float[JaxArray, "D_u"],
    time_step: Scalar,
    state_limits: tuple[Scalar, Scalar],
    velocity_limits: tuple[Scalar, Scalar],
    state_dimension: int,
    rollout_count: int,
) -> Float[JaxArray, "T D_u M"]:
    clipped_controls = jnp.clip(controls, *velocity_limits)
    initial = jnp.broadcast_to(initial_state[:, None], (state_dimension, rollout_count))

    @jax.jit
    @jaxtyped
    def step(
        state: Float[JaxArray, "D_u M"], control: Float[JaxArray, "D_u M"]
    ) -> tuple[Float[JaxArray, "D_u M"], Float[JaxArray, "D_u M"]]:
        new_state = jnp.clip(state + control * time_step, *state_limits)
        return new_state, new_state

    _, states = jax.lax.scan(step, initial, clipped_controls)
    return states


@jax.jit
@jaxtyped
def integrator_step(
    *,
    control: Float[JaxArray, "T D_u"],
    state: Float[JaxArray, "D_u"],
    time_step: Scalar,
    state_limits: tuple[Scalar, Scalar],
    velocity_limits: tuple[Scalar, Scalar],
) -> Float[JaxArray, "D_u"]:
    clipped_control = jnp.clip(control[0], *velocity_limits)
    return jnp.clip(state + clipped_control * time_step, *state_limits)
