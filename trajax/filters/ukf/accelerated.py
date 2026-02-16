from typing import Protocol, NamedTuple, runtime_checkable

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


class JaxUnscentedKalmanFilter(NamedTuple):
    """Unscented Kalman Filter for nonlinear systems with linear observations."""

    alpha: float
    beta: float

    @staticmethod
    def create(*, alpha: float = 1.0, beta: float = 2.0) -> "JaxUnscentedKalmanFilter":
        """Create an Unscented Kalman Filter.

        Args:
            alpha: Controls spread of sigma points - λ = (α² - 1)n.
            beta: Incorporates prior knowledge (2 is used for Gaussian).
        """
        return JaxUnscentedKalmanFilter(
            alpha=alpha,
            beta=beta,
        )

    @jax.jit
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
                initial_state_covariance=initial_state_covariance,
            )
            belief = self.update(
                observation=observation,
                prediction=belief,
                observation_matrix=observation_matrix,
                observation_noise_covariance=observation_noise_covariance,
                initial_state_covariance=initial_state_covariance,
            )

        return belief

    @jax.jit
    @jaxtyped
    def predict(
        self,
        *,
        belief: JaxGaussianBelief,
        state_transition: StateTransitionFunction,
        process_noise_covariance: Float[JaxArray, "D_x D_x"],
        initial_state_covariance: Float[JaxArray, "D_x D_x"],
    ) -> JaxGaussianBelief:
        """Performs the prediction step of the UKF using the unscented transform.

        Args:
            belief: The current belief about the state.
            state_transition: Nonlinear state transition function.
            process_noise_covariance: R matrix representing the covariance of process noise.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
        """
        R = process_noise_covariance
        mu, sigma = belief

        state_dimension = mu.shape[0]
        lambda_ = self.scaling_parameter_for(state_dimension)

        mean_weights, covariance_weights = self._compute_weights(state_dimension)

        def should_skip(
            covariance: Float[JaxArray, "D_x D_x K"],
        ) -> Float[JaxArray, "K"]:
            return jnp.any(jnp.isnan(covariance), axis=(0, 1))

        def substitute_missing_values(
            *,
            mean: Float[JaxArray, "D_x K"],
            covariance: Float[JaxArray, "D_x D_x K"],
        ) -> tuple[Float[JaxArray, "D_x K"], Float[JaxArray, "D_x D_x K"]]:
            covariance_is_nan = should_skip(covariance)

            return (
                jnp.where(jnp.isnan(mean), 0.0, mean),
                jnp.where(
                    covariance_is_nan[jnp.newaxis, jnp.newaxis, :],
                    initial_state_covariance[..., jnp.newaxis],
                    covariance,
                ),
            )

        def predict_single(
            mu_k: JaxArray, sigma_k: JaxArray
        ) -> tuple[JaxArray, JaxArray]:
            sigma_points = self._generate_sigma_points(
                mu_k, sigma_k, state_dimension, lambda_
            )

            def propagate_sigma_point(sigma_point: JaxArray) -> JaxArray:
                return state_transition(sigma_point[:, jnp.newaxis]).flatten()

            propagated_sigma_points = jax.vmap(propagate_sigma_point)(sigma_points)

            predicted_mean_k = jnp.sum(
                mean_weights[:, jnp.newaxis] * propagated_sigma_points, axis=0
            )

            deviations = propagated_sigma_points - predicted_mean_k
            predicted_covariance_k = (
                jnp.einsum("i,ij,ik->jk", covariance_weights, deviations, deviations)
                + R
            )

            return predicted_mean_k, predicted_covariance_k

        def restore_missing(
            *,
            skip_mask: Float[JaxArray, "K"],
            predicted_mean: Float[JaxArray, "D_x K"],
            predicted_covariance: Float[JaxArray, "D_x D_x K"],
        ) -> JaxGaussianBelief:
            return JaxGaussianBelief(
                mean=jnp.where(skip_mask, jnp.nan, predicted_mean),
                covariance=jnp.where(
                    skip_mask[jnp.newaxis, jnp.newaxis, :],
                    jnp.nan,
                    predicted_covariance,
                ),
            )

        skip_mask = should_skip(sigma)
        safe_mu, safe_sigma = substitute_missing_values(mean=mu, covariance=sigma)

        predicted_means, predicted_covariances = jax.vmap(
            predict_single, in_axes=(1, 2), out_axes=(1, 2)
        )(safe_mu, safe_sigma)

        return restore_missing(
            skip_mask=skip_mask,
            predicted_mean=predicted_means,
            predicted_covariance=predicted_covariances,
        )

    @jax.jit
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
        """Performs the update step of the UKF using a linear observation model.

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

    @jax.jit
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

    def scaling_parameter_for(self, state_dimension: int) -> float:
        """Returns the scaling parameter λ = (α² - 1)n for the given state dimension."""
        return (self.alpha**2 - 1) * state_dimension

    def _compute_weights(self, state_dimension: int) -> tuple[JaxArray, JaxArray]:
        n = state_dimension
        sigma_point_count = 2 * n + 1
        lambda_ = self.scaling_parameter_for(state_dimension)
        alpha = self.alpha
        beta = self.beta

        mean_weights = jnp.zeros(sigma_point_count)
        covariance_weights = jnp.zeros(sigma_point_count)

        mean_weights = mean_weights.at[0].set(lambda_ / (n + lambda_))
        covariance_weights = covariance_weights.at[0].set(
            lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        )

        remaining_weight = 1 / (2 * (n + lambda_))
        mean_weights = mean_weights.at[1:].set(remaining_weight)
        covariance_weights = covariance_weights.at[1:].set(remaining_weight)

        return mean_weights, covariance_weights

    @staticmethod
    def _generate_sigma_points(
        mean: JaxArray, covariance: JaxArray, state_dimension: int, lambda_: float
    ) -> JaxArray:
        n = state_dimension
        sigma_point_count = 2 * n + 1

        scaled_covariance = (n + lambda_) * covariance

        # NOTE: This is to ensure the scaled covariance is positive definite.
        scaled_covariance = (scaled_covariance + scaled_covariance.T) / 2
        eigenvalues, eigenvectors = jnp.linalg.eigh(scaled_covariance)
        eigenvalues = jnp.maximum(eigenvalues, 1e-10)
        scaled_covariance = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T

        sqrt_scaled_covariance = jnp.linalg.cholesky(scaled_covariance)

        sigma_points = jnp.zeros((sigma_point_count, n))
        sigma_points = sigma_points.at[0].set(mean)

        for i in range(n):
            sigma_points = sigma_points.at[i + 1].set(
                mean + sqrt_scaled_covariance[:, i]
            )
            sigma_points = sigma_points.at[n + i + 1].set(
                mean - sqrt_scaled_covariance[:, i]
            )

        return sigma_points
