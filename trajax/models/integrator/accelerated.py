from typing import Final, cast
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    DynamicalModel,
    ObstacleModel,
    JaxIntegratorState,
    JaxIntegratorStateSequence,
    JaxIntegratorStateBatch,
    JaxIntegratorControlInputSequence,
    JaxIntegratorControlInputBatch,
    JaxIntegratorObstacleStatesHistory,
    EstimatedObstacleStates,
)
from trajax.states import (
    JaxSimpleState as SimpleState,
    JaxSimpleStateSequence as SimpleStateSequence,
    JaxSimpleStateBatch as SimpleStateBatch,
    JaxSimpleControlInputBatch as SimpleControlInputBatch,
)

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


NO_LIMITS: Final = (jnp.asarray(-jnp.inf), jnp.asarray(jnp.inf))


@jaxtyped
@dataclass(frozen=True)
class JaxIntegratorObstacleStates[D_o: int, K: int]:
    array: Float[JaxArray, "D_o K"]

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[1])


@jaxtyped
@dataclass(frozen=True)
class JaxIntegratorObstacleStateSequences[T: int, D_o: int, K: int]:
    array: Float[JaxArray, "T D_o K"]

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[1])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[2])


@jaxtyped
@dataclass(frozen=True)
class JaxIntegratorObstacleVelocities[D_o: int, K: int]:
    array: Float[JaxArray, "D_o K"]

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[1])


@jaxtyped
@dataclass(frozen=True)
class JaxIntegratorObstacleControlInputSequences[T: int, D_o: int, K: int]:
    array: Float[JaxArray, "T D_o K"]

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[1])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[2])


@dataclass(kw_only=True, frozen=True)
class JaxIntegratorModel(
    DynamicalModel[
        JaxIntegratorState,
        JaxIntegratorStateSequence,
        JaxIntegratorStateBatch,
        JaxIntegratorControlInputSequence,
        JaxIntegratorControlInputBatch,
    ]
):
    _time_step_size: float
    time_step_size_scalar: Scalar
    state_limits: tuple[Scalar, Scalar]
    velocity_limits: tuple[Scalar, Scalar]

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
            _time_step_size=time_step_size,
            time_step_size_scalar=jnp.asarray(time_step_size),
            state_limits=wrap(state_limits) if state_limits is not None else NO_LIMITS,
            velocity_limits=wrap(velocity_limits)
            if velocity_limits is not None
            else NO_LIMITS,
        )

    def simulate[T: int, D_u: int, D_x: int, M: int](
        self,
        inputs: JaxIntegratorControlInputBatch[T, D_u, M],
        initial_state: JaxIntegratorState[D_x],
    ) -> SimpleStateBatch[T, D_x, M]:
        initial = jnp.broadcast_to(
            initial_state.array[:, None],
            (initial_state.dimension, inputs.rollout_count),
        )

        return SimpleStateBatch(
            simulate(
                controls=inputs.array,
                initial_state=initial,
                time_step=self.time_step_size_scalar,
                state_limits=self.state_limits,
                velocity_limits=self.velocity_limits,
            )
        )

    def step[T: int, D_u: int, D_x: int](
        self,
        inputs: JaxIntegratorControlInputSequence[T, D_u],
        state: JaxIntegratorState[D_x],
    ) -> SimpleState[D_x]:
        return SimpleState(
            step(
                control=inputs.array,
                state=state.array,
                time_step=self.time_step_size_scalar,
                state_limits=self.state_limits,
                velocity_limits=self.velocity_limits,
            )
        )

    def forward[T: int, D_x: int](
        self,
        inputs: JaxIntegratorControlInputSequence[T, D_x],
        state: JaxIntegratorState[D_x],
    ) -> SimpleStateSequence[T, D_x]:
        return self.simulate(
            inputs=SimpleControlInputBatch.of(inputs), initial_state=state
        ).rollout(0)

    @property
    def time_step_size(self) -> float:
        return self._time_step_size


@dataclass(kw_only=True, frozen=True)
class JaxIntegratorObstacleModel(
    ObstacleModel[
        JaxIntegratorObstacleStatesHistory,
        JaxIntegratorObstacleStates,
        JaxIntegratorObstacleVelocities,
        JaxIntegratorObstacleControlInputSequences,
        JaxIntegratorObstacleStateSequences,
    ]
):
    time_step: Scalar

    @staticmethod
    def create(*, time_step_size: float) -> "JaxIntegratorObstacleModel":
        """Creates a JAX integrator obstacle model.

        See `JaxIntegratorModel.create` for details on the integrator dynamics.
        """
        return JaxIntegratorObstacleModel(time_step=jnp.asarray(time_step_size))

    def estimate_state_from[D_o: int, K: int](
        self, history: JaxIntegratorObstacleStatesHistory[int, D_o, K]
    ) -> EstimatedObstacleStates[
        JaxIntegratorObstacleStates[D_o, K], JaxIntegratorObstacleVelocities[D_o, K]
    ]:
        assert history.horizon > 0, "History must have at least one time step."

        velocities = estimate_velocities(
            history=history.array, time_step=self.time_step
        )

        return EstimatedObstacleStates(
            states=JaxIntegratorObstacleStates(history.array[-1, :, :]),
            velocities=JaxIntegratorObstacleVelocities(velocities),
        )

    def input_to_maintain[D_o: int, K: int](
        self,
        velocities: JaxIntegratorObstacleVelocities[D_o, K],
        *,
        states: JaxIntegratorObstacleStates[D_o, K],
        horizon: int,
    ) -> JaxIntegratorObstacleControlInputSequences[int, D_o, K]:
        return JaxIntegratorObstacleControlInputSequences(
            jnp.tile(velocities.array[jnp.newaxis, :, :], (horizon, 1, 1))
        )

    def forward[T: int, D_o: int, K: int](
        self,
        *,
        current: JaxIntegratorObstacleStates[D_o, K],
        inputs: JaxIntegratorObstacleControlInputSequences[T, D_o, K],
    ) -> JaxIntegratorObstacleStateSequences[T, D_o, K]:
        result = simulate(
            controls=inputs.array,
            initial_state=current.array,
            time_step=self.time_step,
            state_limits=NO_LIMITS,
            velocity_limits=NO_LIMITS,
        )
        return JaxIntegratorObstacleStateSequences(result)


def wrap(limits: tuple[float, float]) -> tuple[Scalar, Scalar]:
    return (jnp.asarray(limits[0]), jnp.asarray(limits[1]))


@jax.jit
@jaxtyped
def simulate(
    *,
    controls: Float[JaxArray, "T D_u N"],
    initial_state: Float[JaxArray, "D_u N"],
    time_step: Scalar,
    state_limits: tuple[Scalar, Scalar],
    velocity_limits: tuple[Scalar, Scalar],
) -> Float[JaxArray, "T D_u N"]:
    clipped_controls = jnp.clip(controls, *velocity_limits)

    @jaxtyped
    def step(
        state: Float[JaxArray, "D_u N"], control: Float[JaxArray, "D_u N"]
    ) -> tuple[Float[JaxArray, "D_u N"], Float[JaxArray, "D_u N"]]:
        new_state = jnp.clip(state + control * time_step, *state_limits)
        return new_state, new_state

    _, states = jax.lax.scan(step, initial_state, clipped_controls)
    return states


@jax.jit
@jaxtyped
def step(
    *,
    control: Float[JaxArray, "T D_u"],
    state: Float[JaxArray, "D_u"],
    time_step: Scalar,
    state_limits: tuple[Scalar, Scalar],
    velocity_limits: tuple[Scalar, Scalar],
) -> Float[JaxArray, "D_u"]:
    clipped_control = jnp.clip(control[0], *velocity_limits)
    return jnp.clip(state + clipped_control * time_step, *state_limits)


@jax.jit
@jaxtyped
def estimate_velocities(
    *, history: Float[JaxArray, "T D_o K"], time_step: Scalar
) -> Float[JaxArray, "D_o K"]:
    return jax.lax.cond(
        history.shape[0] > 1,
        lambda: (history[-1] - history[-2]) / time_step,
        lambda: jnp.zeros_like(history[-1]),
    )
