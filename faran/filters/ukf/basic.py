from typing import Protocol, NamedTuple, runtime_checkable, cast

from faran.filters.kf import NumPyGaussianBelief, numpy_kalman_filter

from numtypes import Dims, Array, IndexArray, shape_of

import numpy as np


@runtime_checkable
class StateTransitionFunction[D_x: int, K: int](Protocol):
    def __call__(self, state: Array[Dims[D_x, K]]) -> Array[Dims[D_x, K]]:
        """Applies the state transition function to the state."""
        ...


class NumPyUnscentedKalmanFilter(NamedTuple):
    """Unscented Kalman Filter for nonlinear systems with linear observations."""

    alpha: float
    beta: float

    @staticmethod
    def create(
        *, alpha: float = 1.0, beta: float = 2.0
    ) -> "NumPyUnscentedKalmanFilter":
        """Create an Unscented Kalman Filter.

        Args:
            alpha: Controls spread of sigma points - λ = (α² - 1)n.
            beta: Incorporates prior knowledge (2 is optimal for Gaussian).
        """
        return NumPyUnscentedKalmanFilter(alpha=alpha, beta=beta)

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
        """Performs the prediction step of the UKF using the unscented transform.

        Args:
            belief: The current belief about the state.
            state_transition: Nonlinear state transition function.
            process_noise_covariance: R matrix representing the covariance of process noise.
        """
        R = process_noise_covariance
        mu, sigma = belief

        n, obstacle_count = mu.shape
        lambda_ = self.scaling_parameter_for(n)
        sigma_point_count = 2 * n + 1

        mean_weights, covariance_weights = self._compute_weights(n, lambda_)

        def propagate[N_SigmaPoints: int](
            sigma_points: Array[Dims[K, N_SigmaPoints, D_x]],
        ) -> Array[Dims[K, N_SigmaPoints, D_x]]:
            all_points = sigma_points.reshape(-1, n).T
            propagated_flat = state_transition(all_points)  # type: ignore
            return propagated_flat.T.reshape(-1, sigma_point_count, n)  # type: ignore

        def reconstruct_belief[N_SigmaPoints: int](
            propagated: Array[Dims[K, N_SigmaPoints, D_x]],
        ) -> tuple[Array[Dims[K, D_x]], Array[Dims[K, D_x, D_x]]]:
            means = np.sum(
                mean_weights[np.newaxis, :, np.newaxis] * propagated,
                axis=1,
            )

            deviations = propagated - means[:, np.newaxis, :]
            covariances = (
                np.einsum("s,ksi,ksj->kij", covariance_weights, deviations, deviations)
                + R
            )

            return means, covariances

        def scatter_to_full[K_valid: int](
            *,
            means: Array[Dims[K_valid, D_x]],
            covariances: Array[Dims[K_valid, D_x, D_x]],
            valid_indices: IndexArray[Dims[K_valid]],
        ) -> NumPyGaussianBelief[D_x, K]:
            predicted_mean = np.full_like(mu, np.nan)
            predicted_covariance = np.full_like(sigma, np.nan)
            predicted_mean[:, valid_indices] = means.T  # type: ignore
            predicted_covariance[:, :, valid_indices] = covariances.transpose(1, 2, 0)  # type: ignore
            return NumPyGaussianBelief(
                mean=predicted_mean, covariance=predicted_covariance
            )

        valid_indices = np.where(~np.any(np.isnan(sigma), axis=(0, 1)))[0]

        if len(valid_indices) == 0:
            return NumPyGaussianBelief(
                mean=np.full_like(mu, np.nan),
                covariance=np.full_like(sigma, np.nan),
            )

        sigma_points = self._generate_sigma_points_batch(
            means=mu[:, valid_indices],
            covariances=sigma[:, :, valid_indices],
            lambda_=lambda_,
        )
        propagated = propagate(sigma_points)
        means, covariances = reconstruct_belief(propagated)

        return scatter_to_full(
            means=means, covariances=covariances, valid_indices=valid_indices
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
        """Performs the update step of the UKF using a linear observation model.

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

    def scaling_parameter_for(self, state_dimension: int) -> float:
        """Returns the scaling parameter λ = (α² - 1)n for the given state dimension."""
        return (self.alpha**2 - 1) * state_dimension

    def _compute_weights[D_x: int = int, N_SigmaPoints: int = int](
        self, state_dimension: D_x, lambda_: float
    ) -> tuple[Array[Dims[N_SigmaPoints]], Array[Dims[N_SigmaPoints]]]:
        n = state_dimension
        sigma_point_count = cast(N_SigmaPoints, 2 * n + 1)

        mean_weights = np.zeros(sigma_point_count)
        covariance_weights = np.zeros(sigma_point_count)

        mean_weights[0] = lambda_ / (n + lambda_)
        covariance_weights[0] = lambda_ / (n + lambda_) + (
            1 - self.alpha**2 + self.beta
        )

        remaining_weight = 1 / (2 * (n + lambda_))
        mean_weights[1:] = remaining_weight
        covariance_weights[1:] = remaining_weight

        assert shape_of(mean_weights, matches=(sigma_point_count,))
        assert shape_of(covariance_weights, matches=(sigma_point_count,))

        return mean_weights, covariance_weights

    def _generate_sigma_points_batch[D_x: int, K: int, N_SigmaPoints: int = int](
        self,
        *,
        means: Array[Dims[D_x, K]],
        covariances: Array[Dims[D_x, D_x, K]],
        lambda_: float,
    ) -> Array[Dims[K, N_SigmaPoints, D_x]]:
        n, obstacle_count = means.shape
        sigma_point_count = cast(N_SigmaPoints, 2 * n + 1)

        batched_covariance = covariances.transpose(2, 0, 1)
        scaled = (n + lambda_) * batched_covariance

        # NOTE: To ensure symmetric positive definite
        scaled = (scaled + scaled.transpose(0, 2, 1)) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(scaled)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        scaled = eigenvectors @ (
            eigenvalues[..., np.newaxis] * eigenvectors.transpose(0, 2, 1)
        )

        sqrt_covariance = np.linalg.cholesky(scaled)
        means_t = means.T

        # NOTE: Use columns of L (transposed to row-major) for sigma point offsets
        sqrt_covariance_columns = sqrt_covariance.transpose(0, 2, 1)

        sigma_points = np.empty((obstacle_count, sigma_point_count, n))
        sigma_points[:, 0, :] = means_t
        sigma_points[:, 1 : n + 1, :] = (
            means_t[:, np.newaxis, :] + sqrt_covariance_columns
        )
        sigma_points[:, n + 1 :, :] = (
            means_t[:, np.newaxis, :] - sqrt_covariance_columns
        )

        assert shape_of(sigma_points, matches=(obstacle_count, sigma_point_count, n))

        return sigma_points
