from typing import Protocol, runtime_checkable
from dataclasses import dataclass

from faran.types import (
    Array,
    NumPyGaussianBelief,
    NumPyNoiseCovariances,
    NumPyNoiseModelProvider,
)
from faran.filters.noise import IdentityNoiseModelProvider
from faran.filters.kf import numpy_kalman_filter

from jaxtyping import Float

import numpy as np


@runtime_checkable
class StateTransitionFunction(Protocol):
    def __call__(self, state: Float[Array, "D_x K"]) -> Float[Array, "D_x K"]:
        """Applies the state transition function to the state."""
        ...

    def jacobian(self, state: Float[Array, "D_x K"]) -> Float[Array, "D_x D_x K"]:
        """Returns the Jacobian of the state transition function at the given state."""
        ...


@dataclass(frozen=True)
class NumPyExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear systems with linear observations."""

    noise_model: NumPyNoiseModelProvider

    @staticmethod
    def create(
        *, noise_model: NumPyNoiseModelProvider | None = None
    ) -> "NumPyExtendedKalmanFilter":
        return NumPyExtendedKalmanFilter(
            noise_model=IdentityNoiseModelProvider()
            if noise_model is None
            else noise_model
        )

    def filter(
        self,
        observations: Float[Array, "T D_z K"],
        *,
        initial_state_covariance: Float[Array, "D_x D_x"],
        state_transition: StateTransitionFunction,
        process_noise_covariance: Float[Array, "D_x D_x"],
        observation_noise_covariance: Float[Array, "D_z D_z"],
        observation_matrix: Float[Array, "D_z D_x"],
    ) -> NumPyGaussianBelief:
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
        noise = NumPyNoiseCovariances(
            process_noise_covariance, observation_noise_covariance
        )
        adapt = self.noise_model(observation_matrix=observation_matrix, noise=noise)
        noise_state = adapt.state

        for observation in observations:
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
                observation=observation,
                prediction=belief,
                observation_matrix=observation_matrix,
                observation_noise_covariance=noise.observation_noise_covariance,
                initial_state_covariance=initial_state_covariance,
            )

        return belief

    def predict(
        self,
        *,
        belief: NumPyGaussianBelief,
        state_transition: StateTransitionFunction,
        process_noise_covariance: Float[Array, "D_x D_x"],
    ) -> NumPyGaussianBelief:
        """Performs the prediction step of the EKF from the existing belief.

        Args:
            belief: The current belief about the state.
            state_transition: Nonlinear differentiable state transition function.
            process_noise_covariance: R matrix representing the covariance of process noise.
        """
        R = process_noise_covariance
        mu, sigma = belief

        F = state_transition.jacobian(mu)

        F_t = F.transpose(2, 0, 1)
        sigma_t = sigma.transpose(2, 0, 1)
        predicted_covariance = (F_t @ sigma_t @ F_t.transpose(0, 2, 1)).transpose(
            1, 2, 0
        ) + R[..., np.newaxis]

        return NumPyGaussianBelief(
            mean=state_transition(mu), covariance=predicted_covariance
        )

    def update(
        self,
        observation: Float[Array, "D_z K"],
        *,
        prediction: NumPyGaussianBelief,
        observation_matrix: Float[Array, "D_z D_x"],
        observation_noise_covariance: Float[Array, "D_z D_z"],
        initial_state_covariance: Float[Array, "D_x D_x"],
    ) -> NumPyGaussianBelief:
        """Performs the update step of the EKF using a new observation.

        Args:
            observation: The newly observed state.
            prediction: The predicted belief from the prediction step.
            observation_matrix: H matrix mapping state to observation space.
            observation_noise_covariance: Q matrix representing the covariance of observation noise.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
        """
        return numpy_kalman_filter.update(
            observation=observation,
            prediction=prediction,
            observation_matrix=observation_matrix,
            observation_noise_covariance=observation_noise_covariance,
            initial_state_covariance=initial_state_covariance,
        )

    def initial_belief_from(
        self,
        observations: Float[Array, "T D_z K"],
        *,
        initial_state_covariance: Float[Array, "D_x D_x"],
    ) -> NumPyGaussianBelief:
        """Initializes the belief state from the first observation using a pseudo-inverse.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
        """
        return numpy_kalman_filter.initial_belief_from(
            observations, initial_state_covariance=initial_state_covariance
        )
