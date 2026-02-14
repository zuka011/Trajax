import math
from typing import Final, cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    jaxtyped,
    HasShape,
    DynamicalModel,
    ObstacleModel,
    ObstacleStateEstimator,
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
from trajax.filters import (
    JaxKalmanFilter,
    JaxGaussianBelief,
    JaxNoiseCovarianceArrayDescription,
    JaxNoiseCovarianceDescription,
    jax_kalman_filter,
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

    _array: Float[JaxArray, "D_o K"]
    _covariance: Float[JaxArray, "D_o D_o K"] | None = None
    """State covariance matrix from Kalman filtering. Shape: (D_o, D_o, K)."""

    def __array__(self, dtype: None | type = None) -> Array[Dims[D_o, K]]:
        return self._numpy_array

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[1])

    @property
    def array(self) -> Float[JaxArray, "D_o K"]:
        return self._array

    @property
    def covariance(self) -> Float[JaxArray, "D_o D_o K"] | None:
        return self._covariance

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
    _covariance: Float[JaxArray, "D_o D_o K"] | None = None
    """Input covariance matrix from Kalman filtering. Shape: (D_o, D_o, K)."""

    def __array__(self, dtype: None | type = None) -> Array[Dims[D_o, K]]:
        return self._numpy_array

    def zeroed(self, *, at: tuple[int, ...]) -> "JaxIntegratorObstacleInputs[D_o, K]":
        """Returns new obstacle inputs with inputs at specified state dimensions zeroed out."""

        return JaxIntegratorObstacleInputs(self.array.at[at].set(0.0), self._covariance)

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[1])

    @property
    def covariance(self) -> Float[JaxArray, "D_o D_o K"] | None:
        return self._covariance

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

    def forward[T: int, D_o: int, K: int](
        self,
        *,
        current: JaxIntegratorObstacleStates[D_o, K],
        inputs: JaxIntegratorObstacleInputs[D_o, K],
        horizon: T,
    ) -> JaxIntegratorObstacleStateSequences[T, D_o, K]:
        input_sequences = self._input_to_maintain(inputs, horizon=horizon)

        result = simulate(
            controls=input_sequences.array,
            initial_state=current.array,
            time_step=self.time_step,
            state_limits=NO_LIMITS,
            velocity_limits=NO_LIMITS,
        )
        return JaxIntegratorObstacleStateSequences(result)

    def state_jacobian[T: int, D_o: int, K: int](
        self,
        *,
        states: JaxIntegratorObstacleStateSequences[T, D_o, K],
        inputs: JaxIntegratorObstacleInputs[D_o, K],
    ) -> Float[JaxArray, "T D_o D_o K"]:
        raise NotImplementedError(
            "State Jacobian is not implemented for JaxIntegratorObstacleModel."
        )

    def input_jacobian[T: int, D_o: int, K: int](
        self,
        *,
        states: JaxIntegratorObstacleStateSequences[T, D_o, K],
        inputs: JaxIntegratorObstacleInputs[D_o, K],
    ) -> Float[JaxArray, "T D_o D_o K"]:
        raise NotImplementedError(
            "Input Jacobian is not implemented for JaxIntegratorObstacleModel."
        )

    def _input_to_maintain[T: int, D_o: int, K: int](
        self, inputs: JaxIntegratorObstacleInputs[D_o, K], *, horizon: T
    ) -> JaxIntegratorObstacleControlInputSequences[T, D_o, K]:
        return JaxIntegratorObstacleControlInputSequences(
            jnp.tile(inputs.array[jnp.newaxis, :, :], (horizon, 1, 1))
        )


@dataclass(frozen=True)
class JaxFiniteDifferenceIntegratorStateEstimator(
    ObstacleStateEstimator[
        JaxIntegratorObstacleStatesHistory,
        JaxIntegratorObstacleStates,
        JaxIntegratorObstacleInputs,
    ]
):
    time_step_size: Scalar

    @staticmethod
    def create(
        *, time_step_size: float
    ) -> "JaxFiniteDifferenceIntegratorStateEstimator":
        return JaxFiniteDifferenceIntegratorStateEstimator(
            time_step_size=jnp.asarray(time_step_size)
        )

    def estimate_from[D_o: int, K: int, T: int = int](
        self, history: JaxIntegratorObstacleStatesHistory[T, D_o, K]
    ) -> EstimatedObstacleStates[
        JaxIntegratorObstacleStates[D_o, K], JaxIntegratorObstacleInputs[D_o, K], None
    ]:
        velocities = self.estimate_velocities_from(history)

        return EstimatedObstacleStates(
            states=JaxIntegratorObstacleStates(history.array[-1, :, :]),
            inputs=JaxIntegratorObstacleInputs(velocities),
            covariance=None,
        )

    def estimate_velocities_from[D_o: int, K: int, T: int = int](
        self, history: JaxIntegratorObstacleStatesHistory[T, D_o, K]
    ) -> Float[JaxArray, "D_o K"]:
        """Estimates velocities from position history using finite differences."""
        return estimate_velocities(history=history.array, time_step=self.time_step_size)


@dataclass(frozen=True)
class JaxKfIntegratorStateEstimator[D_o: int, D_x: int](
    ObstacleStateEstimator[
        JaxIntegratorObstacleStatesHistory,
        JaxIntegratorObstacleStates,
        JaxIntegratorObstacleInputs,
    ]
):
    """Kalman Filter state estimator for integrator model obstacles."""

    time_step_size: float
    process_noise_covariance: Float[JaxArray, "D_x D_x"]
    observation_noise_covariance: Float[JaxArray, "D_o D_o"]
    observation_dimension: D_o
    estimator: JaxKalmanFilter

    @staticmethod
    def create[D_o_: int, D_x_: int](
        *,
        time_step_size: float,
        process_noise_covariance: JaxNoiseCovarianceDescription[D_x_],
        observation_noise_covariance: JaxNoiseCovarianceArrayDescription[D_o_],
        observation_dimension: D_o_ | None = None,
    ) -> "JaxKfIntegratorStateEstimator[D_o_, D_x_]":
        """Creates an integrator state estimator based on the Kalman Filter with the
        specified noise covariances.

        Args:
            time_step_size: The time step size for the integrator.
            process_noise_covariance: The process noise covariance, either as a full
                matrix, a vector of diagonal entries, or a single scalar representing
                isotropic noise across all state dimensions.
            observation_noise_covariance: The observation noise covariance, either as
                a full matrix, a vector of diagonal entries, or a single scalar
                representing isotropic noise across all state dimensions.
            observation_dimension: The observation dimension for the Kalman filter.
                Mandatory if both noise covariances are specified as scalars.
        """
        observation_dimension = observation_dimension_from(
            process_noise_covariance=process_noise_covariance,
            observation_noise_covariance=observation_noise_covariance,
            observation_dimension=observation_dimension,
        )

        return JaxKfIntegratorStateEstimator(
            time_step_size=time_step_size,
            process_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                process_noise_covariance,
                dimension=kf_state_dimension_for(observation_dimension),
            ),
            observation_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                observation_noise_covariance, dimension=observation_dimension
            ),
            observation_dimension=observation_dimension,
            estimator=JaxKalmanFilter.create(),
        )

    def estimate_from[K: int, T: int = int](
        self, history: JaxIntegratorObstacleStatesHistory[T, D_o, K]
    ) -> EstimatedObstacleStates[
        JaxIntegratorObstacleStates[D_o, K],
        JaxIntegratorObstacleInputs[D_o, K],
        Float[JaxArray, "D_x D_x K"],
    ]:
        """Estimate states and velocities using Kalman filtering."""
        assert history.horizon > 0, (
            "History must contain at least one state for estimation."
        )
        assert history.dimension == self.observation_dimension, (
            f"History dimension {history.dimension} does not match expected "
            f"observation dimension {self.observation_dimension}."
        )

        estimate = self.estimator.filter(
            observations=history.array,
            initial_state_covariance=self.initial_state_covariance,
            state_transition_matrix=self.state_transition_matrix,
            process_noise_covariance=self.process_noise_covariance,
            observation_noise_covariance=self.observation_noise_covariance,
            observation_matrix=self.observation_matrix,
        )

        return EstimatedObstacleStates(
            states=self.states_from(estimate),
            inputs=self.inputs_from(estimate),
            covariance=estimate.covariance,
        )

    def states_from[K: int = int](
        self, belief: JaxGaussianBelief
    ) -> JaxIntegratorObstacleStates[D_o, K]:
        D_o = self.observation_dimension

        return JaxIntegratorObstacleStates(
            belief.mean[:D_o, :], belief.covariance[:D_o, :D_o, :]
        )

    def inputs_from[K: int = int](
        self, belief: JaxGaussianBelief
    ) -> JaxIntegratorObstacleInputs[D_o, K]:
        D_o = self.observation_dimension

        return JaxIntegratorObstacleInputs(
            belief.mean[D_o:, :], belief.covariance[D_o:, D_o:, :]
        )

    @cached_property
    def initial_state_covariance(self) -> Float[JaxArray, "D_x D_x"]:
        D_o = self.observation_dimension

        # NOTE: We are sure of the observed states, unsure of the velocities.
        return jnp.diag(jnp.concatenate((jnp.full(D_o, 1e-4), jnp.full(D_o, 1e3))))

    @cached_property
    def state_transition_matrix(self) -> Float[JaxArray, "D_x D_x"]:
        D_o = self.observation_dimension

        # NOTE: State transition matrix for constant velocity model.
        return jnp.block(
            [
                [jnp.eye(D_o), self.time_step_size * jnp.eye(D_o)],
                [jnp.zeros((D_o, D_o)), jnp.eye(D_o)],
            ]
        )

    @cached_property
    def observation_matrix(self) -> Float[JaxArray, "D_o D_x"]:
        D_o = self.observation_dimension

        # NOTE: We observe only the positions, not the velocities.
        return jnp.hstack((jnp.eye(D_o), jnp.zeros((D_o, D_o))))


def observation_dimension_from[D_o: int, D_x: int](
    *,
    process_noise_covariance: JaxNoiseCovarianceDescription[D_x],
    observation_noise_covariance: JaxNoiseCovarianceDescription[D_o],
    observation_dimension: D_o | None,
) -> D_o:
    if observation_dimension is not None:
        return observation_dimension

    if isinstance(observation_noise_covariance, HasShape):
        return cast(D_o, observation_noise_covariance.shape[0])

    if isinstance(process_noise_covariance, HasShape):
        return cast(D_o, observation_dimension_for(process_noise_covariance.shape[0]))

    assert False, (
        "Observation dimension must be specified if noise covariances are not provided as arrays."
    )


def kf_state_dimension_for(observation_dimension: int) -> int:
    # NOTE: For the integrator model, both the states and state velocities are combined
    # into the estimated state vector.
    return 2 * observation_dimension


def observation_dimension_for(kf_state_dimension: int) -> int:
    # NOTE: For the integrator model, both the states and state velocities are combined
    # into the estimated state vector.
    return kf_state_dimension // 2


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
