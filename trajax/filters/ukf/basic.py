from typing import Protocol, runtime_checkable
from dataclasses import dataclass

from numtypes import Dims, Array

import numpy as np
from scipy.linalg import cholesky

from trajax.filters.kf import NumPyGaussianBelief, numpy_kalman_filter


@runtime_checkable
class StateTransitionFunction[D_x: int, K: int](Protocol):
    def __call__(self, state: Array[Dims[D_x, K]]) -> Array[Dims[D_x, K]]:
        """Applies the state transition function to the state."""
        ...


@dataclass(kw_only=True)
class NumPyUnscentedKalmanFilter:
    """Unscented Kalman Filter for nonlinear systems with linear observations."""

    alpha: float = 1.0
    beta: float = 2.0

    @staticmethod
    def create(
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
    ) -> "NumPyUnscentedKalmanFilter":
        """Create an Unscented Kalman Filter.

        Args:
            alpha: Controls spread of sigma points - λ = (α² - 1)n.
            beta: Incorporates prior knowledge (2 is optimal for Gaussian).
        """
        return NumPyUnscentedKalmanFilter(
            alpha=alpha,
            beta=beta,
        )

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
        """Run the UKF over a sequence of observations.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Σ₀ matrix representing initial state uncertainty.
            state_transition: Nonlinear state transition function.
            process_noise_covariance: R matrix representing the covariance of process noise.
            observation_noise_covariance: Q matrix representing the covariance of observation noise.
            observation_matrix: H matrix mapping state to observation space.
        """
        belief = self.initial_belief_from(
            observations,
            initial_state_covariance=initial_state_covariance,
            observation_matrix=observation_matrix,
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
            )

        return belief

    def predict[D_x: int, K: int](
        self,
        *,
        belief: NumPyGaussianBelief[D_x, K],
        state_transition: StateTransitionFunction[D_x, K],
        process_noise_covariance: Array[Dims[D_x, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
        """Performs the prediction step of the UKF using the unscented transform.

        Args:
            belief: The current belief about the state.
            state_transition: Nonlinear state transition function.
            process_noise_covariance: R matrix representing the covariance of process noise.
        """
        R = process_noise_covariance
        mu, sigma = belief

        state_dimension = mu.shape[0]
        batch_count = mu.shape[1]
        lambda_ = self.scaling_parameter_for(state_dimension)

        mean_weights, covariance_weights = self._compute_weights(
            state_dimension, lambda_
        )

        predicted_mean = np.zeros_like(mu)
        predicted_covariance = np.zeros_like(sigma)

        for k in range(batch_count):
            mu_k = mu[:, k]
            sigma_k = sigma[:, :, k]

            sigma_points = self._generate_sigma_points(
                mu_k, sigma_k, state_dimension, lambda_
            )
            propagated_sigma_points = np.array(
                [state_transition(sp[:, np.newaxis]).flatten() for sp in sigma_points]
            )

            predicted_mean_k = np.sum(
                mean_weights[:, np.newaxis] * propagated_sigma_points, axis=0
            )

            deviations = propagated_sigma_points - predicted_mean_k
            predicted_covariance_k = (
                np.einsum("i,ij,ik->jk", covariance_weights, deviations, deviations) + R
            )

            predicted_mean[:, k] = predicted_mean_k
            predicted_covariance[:, :, k] = predicted_covariance_k

        return NumPyGaussianBelief(mean=predicted_mean, covariance=predicted_covariance)

    def update[D_x: int, D_z: int, K: int](
        self,
        observation: Array[Dims[D_z, K]],
        *,
        prediction: NumPyGaussianBelief[D_x, K],
        observation_matrix: Array[Dims[D_z, D_x]],
        observation_noise_covariance: Array[Dims[D_z, D_z]],
    ) -> NumPyGaussianBelief[D_x, K]:
        """Performs the update step of the UKF using a linear observation model.

        Args:
            observation: The newly observed state.
            prediction: The predicted belief from the prediction step.
            observation_matrix: H matrix mapping state to observation space.
            observation_noise_covariance: Q matrix representing the covariance of observation noise.
        """
        return numpy_kalman_filter.update(
            observation=observation,
            prediction=prediction,
            observation_matrix=observation_matrix,
            observation_noise_covariance=observation_noise_covariance,
        )

    def initial_belief_from[T: int, D_x: int, D_z: int, K: int](
        self,
        observations: Array[Dims[T, D_z, K]],
        *,
        initial_state_covariance: Array[Dims[D_x, D_x]],
        observation_matrix: Array[Dims[D_z, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
        """Initializes the belief state from the first observation using a pseudo-inverse.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Σ₀ matrix representing initial state uncertainty.
            observation_matrix: H matrix mapping state to observation space.
        """
        return numpy_kalman_filter.initial_belief_from(
            observations,
            initial_state_covariance=initial_state_covariance,
            observation_matrix=observation_matrix,
        )

    def scaling_parameter_for(self, state_dimension: int) -> float:
        """Returns the scaling parameter λ = (α² - 1)n for the given state dimension."""
        return (self.alpha**2 - 1) * state_dimension

    def _compute_weights(
        self, state_dimension: int, lambda_: float
    ) -> tuple[Array, Array]:
        """Compute sigma point weights for mean and covariance.

        Mean weights: w_m,0 = λ/(n+λ), w_m,i = 1/(2(n+λ))
        Covariance weights: w_c,0 = λ/(n+λ) + (1-α²+β), w_c,i = 1/(2(n+λ))
        """
        n = state_dimension
        sigma_point_count = 2 * n + 1

        mean_weights = np.zeros(sigma_point_count)
        covariance_weights = np.zeros(sigma_point_count)

        mean_weights[0] = lambda_ / (n + lambda_)
        covariance_weights[0] = lambda_ / (n + lambda_) + (
            1 - self.alpha**2 + self.beta
        )

        remaining_weight = 1 / (2 * (n + lambda_))
        mean_weights[1:] = remaining_weight
        covariance_weights[1:] = remaining_weight

        return mean_weights, covariance_weights

    def _generate_sigma_points(
        self, mean: Array, covariance: Array, state_dimension: int, lambda_: float
    ) -> Array:
        """Generate 2n+1 sigma points around the mean.

        σ₀ = μ
        σᵢ = μ + (√((n+λ)Σ))ᵢ for i = 1,...,n
        σᵢ = μ - (√((n+λ)Σ))_{i-n} for i = n+1,...,2n
        """
        n = state_dimension
        sigma_point_count = 2 * n + 1

        sigma_points = np.zeros((sigma_point_count, n))
        sigma_points[0] = mean

        scaled_covariance = (n + lambda_) * covariance

        # Ensure the scaled covariance is positive definite.
        scaled_covariance = (scaled_covariance + scaled_covariance.T) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(scaled_covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        scaled_covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        sqrt_scaled_covariance = cholesky(scaled_covariance, lower=True)

        for i in range(n):
            sigma_points[i + 1] = mean + sqrt_scaled_covariance[:, i]
            sigma_points[n + i + 1] = mean - sqrt_scaled_covariance[:, i]

        return sigma_points
