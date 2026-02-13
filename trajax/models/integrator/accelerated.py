import math
from typing import Final, cast
from dataclasses import dataclass
from functools import cached_property

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

from numtypes import Array, Dims
from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp
import numpy as np

NO_LIMITS: Final = (jnp.asarray(-jnp.inf), jnp.asarray(jnp.inf))


@jaxtyped
@dataclass(frozen=True)
class JaxIntegratorObstacleStates[D_o: int, K: int]:
    """Obstacle states represented in integrator model coordinates."""

    array: Float[JaxArray, "D_o K"]

    def __array__(self, dtype: None | type = None) -> Array[Dims[D_o, K]]:
        return self._numpy_array

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[1])

    @cached_property
    def _numpy_array(self) -> Array[Dims[D_o, K]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxIntegratorObstacleStateSequences[T: int, D_o: int, K: int]:
    """Time-indexed obstacle state sequences for integrator model obstacles."""

    array: Float[JaxArray, "T D_o K"]

    def __array__(self, dtype: None | type = None) -> Array[Dims[T, D_o, K]]:
        return self._numpy_array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[1])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[2])

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_o, K]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxIntegratorObstacleInputs[D_o: int, K: int]:
    array: Float[JaxArray, "D_o K"]

    def __array__(self, dtype: None | type = None) -> Array[Dims[D_o, K]]:
        return self._numpy_array

    def zeroed(self, *, at: tuple[int, ...]) -> "JaxIntegratorObstacleInputs[D_o, K]":
        """Returns new obstacle inputs with inputs at specified state dimensions zeroed out."""

        return JaxIntegratorObstacleInputs(self.array.at[at].set(0.0))

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[1])

    @cached_property
    def _numpy_array(self) -> Array[Dims[D_o, K]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxIntegratorObstacleControlInputSequences[T: int, D_o: int, K: int]:
    array: Float[JaxArray, "T D_o K"]

    def __array__(self, dtype: None | type = None) -> Array[Dims[T, D_o, K]]:
        return self._numpy_array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[1])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[2])

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_o, K]]:
        return np.asarray(self.array)


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
    """Point-mass model with direct position control, used for obstacle prediction."""

    _time_step_size: float
    time_step_size_scalar: Scalar
    state_limits: tuple[Scalar, Scalar]
    velocity_limits: tuple[Scalar, Scalar]
    periodic: bool

    @staticmethod
    def create(
        *,
        time_step_size: float,
        state_limits: tuple[float, float] | None = None,
        velocity_limits: tuple[float, float] | None = None,
        periodic: bool = False,
    ) -> "JaxIntegratorModel":
        """A JAX integrator model where state = cumulative sum of controls.

        This model represents a particle that moves according to velocity commands.
        State dimension must equal control dimension (D_x == D_u).

        $$x_{t+1} = \\text{clip}(x_t + \\text{clip}(u_t,\\; v_{\\text{lim}}) \\cdot \\Delta t,\\; s_{\\text{lim}})$$

        Args:
            time_step_size: The time step size for the integrator.
            state_limits: Optional tuple of (min, max) limits for the state values.
            velocity_limits: Optional tuple of (min, max) limits for the velocity inputs.
            periodic: Whether to apply periodic boundary conditions based on state_limits.
        """
        if periodic:
            validate_periodic_state_limits(state_limits)

        return JaxIntegratorModel(
            _time_step_size=time_step_size,
            time_step_size_scalar=jnp.asarray(time_step_size),
            state_limits=wrap(state_limits) if state_limits is not None else NO_LIMITS,
            velocity_limits=wrap(velocity_limits)
            if velocity_limits is not None
            else NO_LIMITS,
            periodic=periodic,
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
            simulate_periodic(
                controls=inputs.array,
                initial_state=initial,
                time_step=self.time_step_size_scalar,
                state_limits=self.state_limits,
                velocity_limits=self.velocity_limits,
            )
            if self.periodic
            else simulate(
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
            step_periodic(
                control=inputs.array,
                state=state.array,
                time_step=self.time_step_size_scalar,
                state_limits=self.state_limits,
                velocity_limits=self.velocity_limits,
            )
            if self.periodic
            else step(
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
        JaxIntegratorObstacleInputs,
        JaxIntegratorObstacleControlInputSequences,
        JaxIntegratorObstacleStateSequences,
    ]
):
    """Propagates integrator dynamics forward with constant velocity."""

    time_step: Scalar

    @staticmethod
    def create(*, time_step_size: float) -> "JaxIntegratorObstacleModel":
        """Creates a JAX integrator obstacle model.

        See `JaxIntegratorModel.create` for details on the integrator dynamics.
        """
        return JaxIntegratorObstacleModel(time_step=jnp.asarray(time_step_size))

    def input_to_maintain[D_o: int, K: int](
        self,
        inputs: JaxIntegratorObstacleInputs[D_o, K],
        *,
        states: JaxIntegratorObstacleStates[D_o, K],
        horizon: int,
    ) -> JaxIntegratorObstacleControlInputSequences[int, D_o, K]:
        return JaxIntegratorObstacleControlInputSequences(
            jnp.tile(inputs.array[jnp.newaxis, :, :], (horizon, 1, 1))
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

    # TODO: Review!
    def state_jacobian[T: int, D_o: int, K: int](
        self,
        *,
        states: JaxIntegratorObstacleStateSequences[T, D_o, K],
        inputs: JaxIntegratorObstacleControlInputSequences[T, D_o, K],
    ) -> Float[JaxArray, "T D_o D_o K"]:
        raise NotImplementedError(
            "State Jacobian is not implemented for JaxIntegratorObstacleModel."
        )

    # TODO: Review!
    def input_jacobian[T: int, D_o: int, K: int](
        self,
        *,
        states: JaxIntegratorObstacleStateSequences[int, D_o, K],
        inputs: JaxIntegratorObstacleControlInputSequences[int, D_o, K],
    ) -> Float[JaxArray, "T D_o D_o K"]:
        raise NotImplementedError(
            "Input Jacobian is not implemented for JaxIntegratorObstacleModel."
        )


@dataclass(frozen=True)
class JaxFiniteDifferenceIntegratorStateEstimator:
    time_step_size: Scalar

    @staticmethod
    def create(
        *, time_step_size: float
    ) -> "JaxFiniteDifferenceIntegratorStateEstimator":
        return JaxFiniteDifferenceIntegratorStateEstimator(
            time_step_size=jnp.asarray(time_step_size)
        )

    def estimate_from[D_o: int, K: int](
        self, history: JaxIntegratorObstacleStatesHistory[int, D_o, K]
    ) -> EstimatedObstacleStates[
        JaxIntegratorObstacleStates[D_o, K], JaxIntegratorObstacleInputs[D_o, K]
    ]:
        velocities = self.estimate_velocities_from(history)

        return EstimatedObstacleStates(
            states=JaxIntegratorObstacleStates(history.array[-1, :, :]),
            inputs=JaxIntegratorObstacleInputs(velocities),
        )

    def estimate_velocities_from[D_o: int, K: int](
        self, history: JaxIntegratorObstacleStatesHistory[int, D_o, K]
    ) -> Float[JaxArray, "D_o K"]:
        """Estimates velocities from position history using finite differences."""
        return estimate_velocities(history=history.array, time_step=self.time_step_size)


def validate_periodic_state_limits(state_limits: tuple[float, float] | None) -> None:
    assert state_limits is not None, (
        "Periodic boundaries require explicit state limits."
    )

    lower, upper = state_limits
    assert math.isfinite(lower) and math.isfinite(upper), (
        "Periodic boundaries must be finite."
    )

    assert upper > lower, (
        "Periodic boundaries require upper limit to be greater than lower limit."
    )


def wrap(limits: tuple[float, float]) -> tuple[Scalar, Scalar]:
    return (jnp.asarray(limits[0]), jnp.asarray(limits[1]))


@jax.jit
@jaxtyped
def wrap_periodic_batch(
    *,
    states: Float[JaxArray, "T D_u N"],
    state_limits: tuple[Scalar, Scalar],
) -> Float[JaxArray, "T D_u N"]:
    lower, upper = state_limits
    period = upper - lower
    wrapped = lower + jnp.mod(states - lower, period)
    return jnp.where(states == upper, upper, wrapped)


@jax.jit
@jaxtyped
def wrap_periodic_state(
    *,
    state: Float[JaxArray, "D_u"],
    state_limits: tuple[Scalar, Scalar],
) -> Float[JaxArray, "D_u"]:
    lower, upper = state_limits
    period = upper - lower
    wrapped = lower + jnp.mod(state - lower, period)
    return jnp.where(state == upper, upper, wrapped)


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
def simulate_periodic(
    *,
    controls: Float[JaxArray, "T D_u N"],
    initial_state: Float[JaxArray, "D_u N"],
    time_step: Scalar,
    state_limits: tuple[Scalar, Scalar],
    velocity_limits: tuple[Scalar, Scalar],
) -> Float[JaxArray, "T D_u N"]:
    clipped_controls = jnp.clip(controls, *velocity_limits)
    unbounded = initial_state + jnp.cumsum(clipped_controls * time_step, axis=0)
    return wrap_periodic_batch(states=unbounded, state_limits=state_limits)


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
def step_periodic(
    *,
    control: Float[JaxArray, "T D_u"],
    state: Float[JaxArray, "D_u"],
    time_step: Scalar,
    state_limits: tuple[Scalar, Scalar],
    velocity_limits: tuple[Scalar, Scalar],
) -> Float[JaxArray, "D_u"]:
    clipped_control = jnp.clip(control[0], *velocity_limits)
    unbounded = state + clipped_control * time_step
    return wrap_periodic_state(state=unbounded, state_limits=state_limits)


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
