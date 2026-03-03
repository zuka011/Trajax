from typing import Protocol, NamedTuple, runtime_checkable

from faran.types import (
    jaxtyped,
    JaxGaussianBelief,
    JaxNoiseCovariances,
    JaxNoiseModelProvider,
)
from faran.filters.noise import IdentityNoiseModelProvider
from faran.filters.kf import jax_kalman_filter

from jaxtyping import Array as JaxArray, Float

import jax
import jax.numpy as jnp


@runtime_checkable
class StateTransitionFunction(Protocol):
    def __call__(self, state: Float[JaxArray, "D_x K"]) -> Float[JaxArray, "D_x K"]:
        """Applies the state transition function to the state."""
        ...

    def jacobian(self, state: Float[JaxArray, "D_x K"]) -> Float[JaxArray, "D_x D_x K"]:
        """Returns the Jacobian of the state transition function at the given state."""
        ...


class JaxExtendedKalmanFilter(NamedTuple):
    """Extended Kalman Filter for nonlinear systems with linear observations."""

    noise_model: JaxNoiseModelProvider

    @staticmethod
    def create(
        *, noise_model: JaxNoiseModelProvider | None = None
    ) -> "JaxExtendedKalmanFilter":
        return JaxExtendedKalmanFilter(
            noise_model=IdentityNoiseModelProvider()
            if noise_model is None
            else noise_model
        )

    @jax.jit(static_argnums=(0,))
    @jaxtyped
    def filter(
        self,
        observations: Float[JaxArray, "T D_z K"],
        *,
        initial_state_covariance: Float[JaxArray, "D_x D_x"],
        state_transition: StateTransitionFunction,
        process_noise_covariance: Float[JaxArray, "D_x D_x"],
        observation_noise_covariance: Float[JaxArray, "D_z D_z"],
        observation_matrix: Float[JaxArray, "D_z D_x"],
    ) -> JaxGaussianBelief:
        """Run the EKF over a sequence of observations.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
            state_transition: Nonlinear differentiable state transition function.
            process_noise_covariance: R matrix representing the covariance of process noise.
            observation_noise_covariance: Q matrix representing the covariance of observation noise.
            observation_matrix: H matrix mapping state to observation space.
        """
        belief = self.initial_belief_from(
            observations, initial_state_covariance=initial_state_covariance
        )
        noise = JaxNoiseCovariances(
            process_noise_covariance, observation_noise_covariance
        )
        adapt = self.noise_model(observation_matrix=observation_matrix, noise=noise)
        noise_state = adapt.state

        def step(carry, observation):
            belief, noise, noise_state = carry
            belief = self.predict(
                belief=belief,
                state_transition=state_transition,
                process_noise_covariance=noise.process_noise_covariance,
            )
            noise, noise_state = adapt(
                noise=noise,
                prediction=belief,
                observation=observation,
                state=noise_state,
            )
            belief = self.update(
                observation,
                prediction=belief,
                observation_matrix=observation_matrix,
                observation_noise_covariance=noise.observation_noise_covariance,
                initial_state_covariance=initial_state_covariance,
            )
            return (belief, noise, noise_state), None

        (belief, _, _), _ = jax.lax.scan(
            step, (belief, noise, noise_state), observations
        )
        return belief

    @jax.jit(static_argnums=(0,))
    @jaxtyped
    def predict(
        self,
        *,
        belief: JaxGaussianBelief,
        state_transition: StateTransitionFunction,
        process_noise_covariance: Float[JaxArray, "D_x D_x"],
    ) -> JaxGaussianBelief:
        """Performs the prediction step of the EKF from the existing belief.

        Args:
            belief: The current belief about the state.
            state_transition: Nonlinear differentiable state transition function.
            process_noise_covariance: R matrix representing the covariance of process noise.
        """
        R = process_noise_covariance
        mu, sigma = belief

        F = state_transition.jacobian(mu)

        return JaxGaussianBelief(
            mean=state_transition(mu),
            covariance=jnp.einsum("ijk,jlk,mlk->imk", F, sigma, F)
            + R[..., jnp.newaxis],
        )

    @jax.jit(static_argnums=(0,))
    @jaxtyped
    def update(
        self,
        observation: Float[JaxArray, "D_z K"],
        *,
        prediction: JaxGaussianBelief,
        observation_matrix: Float[JaxArray, "D_z D_x"],
        observation_noise_covariance: Float[JaxArray, "D_z D_z"],
        initial_state_covariance: Float[JaxArray, "D_x D_x"],
    ) -> JaxGaussianBelief:
        """Performs the update step of the EKF using a new observation.

        Args:
            observation: The newly observed state.
            prediction: The predicted belief from the prediction step.
            observation_matrix: H matrix mapping state to observation space.
            observation_noise_covariance: Q matrix representing the covariance of observation noise.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
        """
        return jax_kalman_filter.update(
            observation=observation,
            prediction=prediction,
            observation_matrix=observation_matrix,
            observation_noise_covariance=observation_noise_covariance,
            initial_state_covariance=initial_state_covariance,
        )

    @jax.jit(static_argnums=(0,))
    @jaxtyped
    def initial_belief_from(
        self,
        observations: Float[JaxArray, "T D_z K"],
        *,
        initial_state_covariance: Float[JaxArray, "D_x D_x"],
    ) -> JaxGaussianBelief:
        """Initializes the belief state from the first observation using a pseudo-inverse.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
        """
        return jax_kalman_filter.initial_belief_from(
            observations, initial_state_covariance=initial_state_covariance
        )
