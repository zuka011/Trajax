from typing import Protocol, runtime_checkable
from dataclasses import dataclass

from numtypes import Dims, Array

import numpy as np

from trajax.filters.kf import NumPyGaussianBelief, numpy_kalman_filter


@runtime_checkable
class StateTransitionFunction[D_x: int, K: int](Protocol):
    def __call__(self, state: Array[Dims[D_x, K]]) -> Array[Dims[D_x, K]]:
        """Applies the state transition function to the state."""
        ...

    def jacobian(self, state: Array[Dims[D_x, K]]) -> Array[Dims[D_x, D_x, K]]:
        """Returns the Jacobian of the state transition function at the given state."""
        ...


@dataclass(kw_only=True)
class NumPyExtendedKalmanFilter:
    """Extended Kalman Filter for nonlinear systems with linear observations."""

    @staticmethod
    def create() -> "NumPyExtendedKalmanFilter":
        return NumPyExtendedKalmanFilter()

    def filter[T: int, D_x: int, D_z: int, K: int](
        self,
        observations: Array[Dims[T, D_z, K]],
        *,
        initial_state_covariance: Array[Dims[D_x, D_x]],
        state_transition: StateTransitionFunction[D_x, K],
        process_noise_covariance: Array[Dims[D_x, D_x]],
        observation_noise_covariance: Array[Dims[D_z, D_z]],
        observation_matrix: Array[Dims[D_z, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
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

        for observation in observations:
            belief = self.predict(
                belief=belief,
                state_transition=state_transition,
                process_noise_covariance=process_noise_covariance,
            )
            belief = self.update(
                observation=observation,
                prediction=belief,
                observation_matrix=observation_matrix,
                observation_noise_covariance=observation_noise_covariance,
                initial_state_covariance=initial_state_covariance,
            )

        return belief

    def predict[D_x: int, K: int](
        self,
        *,
        belief: NumPyGaussianBelief[D_x, K],
        state_transition: StateTransitionFunction[D_x, K],
        process_noise_covariance: Array[Dims[D_x, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
        """Performs the prediction step of the EKF from the existing belief.

        Args:
            belief: The current belief about the state.
            state_transition: Nonlinear differentiable state transition function.
            process_noise_covariance: R matrix representing the covariance of process noise.
        """
        R = process_noise_covariance
        mu, sigma = belief

        F = state_transition.jacobian(mu)

        return NumPyGaussianBelief(
            mean=state_transition(mu),
            covariance=np.einsum("ijk,jlk,mlk->imk", F, sigma, F) + R[..., np.newaxis],
        )

    def update[D_x: int, D_z: int, K: int](
        self,
        observation: Array[Dims[D_z, K]],
        *,
        prediction: NumPyGaussianBelief[D_x, K],
        observation_matrix: Array[Dims[D_z, D_x]],
        observation_noise_covariance: Array[Dims[D_z, D_z]],
        initial_state_covariance: Array[Dims[D_x, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
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

    def initial_belief_from[T: int, D_x: int, D_z: int, K: int](
        self,
        observations: Array[Dims[T, D_z, K]],
        *,
        initial_state_covariance: Array[Dims[D_x, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
        """Initializes the belief state from the first observation using a pseudo-inverse.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
        """
        return numpy_kalman_filter.initial_belief_from(
            observations, initial_state_covariance=initial_state_covariance
        )
