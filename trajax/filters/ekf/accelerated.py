from typing import Protocol, runtime_checkable
from dataclasses import dataclass

from trajax.types import jaxtyped

from jaxtyping import Array as JaxArray, Float

import jax
import jax.numpy as jnp

from trajax.filters.kf import JaxGaussianBelief, jax_kalman_filter


@runtime_checkable
class StateTransitionFunction(Protocol):
    def __call__(self, state: Float[JaxArray, "D_x K"]) -> Float[JaxArray, "D_x K"]:
        """Applies the state transition function to the state."""
        ...

    def jacobian(self, state: Float[JaxArray, "D_x K"]) -> Float[JaxArray, "D_x D_x K"]:
        """Returns the Jacobian of the state transition function at the given state."""
        ...


@dataclass(kw_only=True)
class JaxExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear systems with linear observations."""

    @staticmethod
    def create() -> "JaxExtendedKalmanFilter":
        return JaxExtendedKalmanFilter()

    @staticmethod
    @jax.jit
    @jaxtyped
    def filter(
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
        belief = JaxExtendedKalmanFilter.initial_belief_from(
            observations, initial_state_covariance=initial_state_covariance
        )

        for observation in observations:
            belief = JaxExtendedKalmanFilter.predict(
                belief=belief,
                state_transition=state_transition,
                process_noise_covariance=process_noise_covariance,
            )
            belief = JaxExtendedKalmanFilter.update(
                observation=observation,
                prediction=belief,
                observation_matrix=observation_matrix,
                observation_noise_covariance=observation_noise_covariance,
                initial_state_covariance=initial_state_covariance,
            )

        return belief

    @staticmethod
    @jax.jit
    @jaxtyped
    def predict(
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

    @staticmethod
    @jax.jit
    @jaxtyped
    def update(
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

    @staticmethod
    @jax.jit
    @jaxtyped
    def initial_belief_from(
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
