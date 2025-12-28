from typing import Final
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    DynamicalModel,
    JaxIntegratorState,
    JaxIntegratorStateBatch,
    JaxIntegratorControlInputSequence,
    JaxIntegratorControlInputBatch,
)
from trajax.states import (
    JaxSimpleState as SimpleState,
    JaxSimpleStateBatch as SimpleStateBatch,
)

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


NO_LIMITS: Final = (float("-inf"), float("inf"))


@dataclass(kw_only=True, frozen=True)
class JaxIntegratorModel(
    DynamicalModel[
        JaxIntegratorState,
        JaxIntegratorStateBatch,
        JaxIntegratorControlInputSequence,
        JaxIntegratorControlInputBatch,
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

    def simulate[T: int, D_u: int, D_x: int, M: int](
        self,
        inputs: JaxIntegratorControlInputBatch[T, D_u, M],
        initial_state: JaxIntegratorState[D_x],
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

    def step[T: int, D_u: int, D_x: int](
        self,
        input: JaxIntegratorControlInputSequence[T, D_u],
        state: JaxIntegratorState[D_x],
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
