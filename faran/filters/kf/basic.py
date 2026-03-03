from typing import NamedTuple
from dataclasses import dataclass

from faran.types import (
    Array,
    NumPyGaussianBelief,
    NumPyNoiseCovariances,
    NumPyNoiseModelProvider,
    NumPyNoiseCovarianceDescription,
)
from faran.filters.noise import IdentityNoiseModelProvider

from jaxtyping import Float

import numpy as np


class ObstaclePartitioning(NamedTuple):
    should_update: Float[Array, " K"]
    should_initialize: Float[Array, " K"]


@dataclass(frozen=True)
class NumPyKalmanFilter:
    """Kalman Filter for linear systems."""

    noise_model: NumPyNoiseModelProvider

    @staticmethod
    def create(
        *, noise_model: NumPyNoiseModelProvider | None = None
    ) -> "NumPyKalmanFilter":
        """Creates a Kalman filter for linear systems.

        Args:
            noise_model: An optional noise model provider for adaptive noise estimation.
        """
        return NumPyKalmanFilter(
            noise_model=IdentityNoiseModelProvider()
            if noise_model is None
            else noise_model
        )

    def filter(
        self,
        observations: Float[Array, "T D_z K"],
        *,
        initial_state_covariance: Float[Array, "D_x D_x"],
        state_transition_matrix: Float[Array, "D_x D_x"],
        process_noise_covariance: Float[Array, "D_x D_x"],
        observation_noise_covariance: Float[Array, "D_z D_z"],
        observation_matrix: Float[Array, "D_z D_x"],
    ) -> NumPyGaussianBelief:
        """Performs one step of the Kalman filter given a new observation.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
            state_transition_matrix: A matrix for state prediction.
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
                state_transition_matrix=state_transition_matrix,
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
        state_transition_matrix: Float[Array, "D_x D_x"],
        process_noise_covariance: Float[Array, "D_x D_x"],
    ) -> NumPyGaussianBelief:
        """Performs the prediction step of the Kalman filter from the existing belief.

        Args:
            belief: The current belief about the state.
            state_transition_matrix: A matrix for state prediction.
            process_noise_covariance: R matrix representing the covariance of process noise.
        """
        return numpy_kalman_filter.predict(
            belief=belief,
            state_transition_matrix=state_transition_matrix,
            process_noise_covariance=process_noise_covariance,
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
        """Performs the update step of the Kalman filter using a new observation.

        Args:
            observation: The newly observed state.
            prediction: The predicted belief from the prediction step.
            observation_matrix: H matrix mapping state to observation space.
            observation_noise_covariance: Q matrix representing the covariance of observation noise.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
        """
        return numpy_kalman_filter.update(
            observation,
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

        This method provides a simple way to initialize the state mean based on the first
        observation, while using the provided initial state covariance for uncertainty.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
        """
        return numpy_kalman_filter.initial_belief_from(
            observations, initial_state_covariance=initial_state_covariance
        )


class numpy_kalman_filter:
    @staticmethod
    def predict(
        *,
        belief: NumPyGaussianBelief,
        state_transition_matrix: Float[Array, "D_x D_x"],
        process_noise_covariance: Float[Array, "D_x D_x"],
    ) -> NumPyGaussianBelief:
        A = state_transition_matrix
        R = process_noise_covariance
        mu, sigma = belief

        sigma_t = sigma.transpose(2, 0, 1)
        predicted_covariance = (A @ sigma_t @ A.T).transpose(1, 2, 0) + R[
            ..., np.newaxis
        ]

        return NumPyGaussianBelief(mean=A @ mu, covariance=predicted_covariance)

    @staticmethod
    def update(
        observation: Float[Array, "D_z K"],
        *,
        prediction: NumPyGaussianBelief,
        observation_matrix: Float[Array, "D_z D_x"],
        observation_noise_covariance: Float[Array, "D_z D_z"],
        initial_state_covariance: Float[Array, "D_x D_x"],
    ) -> NumPyGaussianBelief:
        H = observation_matrix
        Q = observation_noise_covariance

        def partition(
            *, observation: Float[Array, "D_z K"], mean: Float[Array, "D_x K"]
        ) -> ObstaclePartitioning:
            observation_valid = ~np.any(np.isnan(observation), axis=0)
            prediction_valid = ~np.any(np.isnan(mean), axis=0)

            return ObstaclePartitioning(
                should_update=observation_valid & prediction_valid,
                should_initialize=observation_valid & ~prediction_valid,
            )

        def substitute_missing_values(
            *,
            observation: Float[Array, "D_z K"],
            mean: Float[Array, "D_x K"],
            covariance: Float[Array, "D_x D_x K"],
        ) -> tuple[
            Float[Array, "D_z K"], Float[Array, "D_x K"], Float[Array, "D_x D_x K"]
        ]:
            # NOTE: We temporarily replace NaNs with zeros to prevent errors.
            covariance_is_nan = np.any(np.isnan(covariance), axis=(0, 1))

            return (
                np.where(np.isnan(observation), 0.0, observation),
                np.where(np.isnan(mean), 0.0, mean),
                np.where(
                    covariance_is_nan[np.newaxis, np.newaxis, :],
                    initial_state_covariance[..., np.newaxis],
                    covariance,
                ),
            )

        def update(
            *,
            observation: Float[Array, "D_z K"],
            mean: Float[Array, "D_x K"],
            covariance: Float[Array, "D_x D_x K"],
        ) -> NumPyGaussianBelief:
            D_x = prediction.mean.shape[0]

            sigma_t = covariance.transpose(2, 0, 1)
            S_t = H @ sigma_t @ H.T + Q
            K_t = np.linalg.solve(S_t, H @ sigma_t).transpose(0, 2, 1)

            innovation = observation - H @ mean
            updated_mean = mean + (K_t @ innovation.T[:, :, np.newaxis]).squeeze(-1).T

            IKH = np.eye(D_x) - K_t @ H
            joseph_term = IKH @ sigma_t @ IKH.transpose(0, 2, 1)
            kalman_noise_term = K_t @ Q @ K_t.transpose(0, 2, 1)
            updated_covariance = (joseph_term + kalman_noise_term).transpose(1, 2, 0)

            return NumPyGaussianBelief(mean=updated_mean, covariance=updated_covariance)

        def initialize_from(observation: Float[Array, "D_z K"]) -> NumPyGaussianBelief:
            K = observation.shape[1]

            # NOTE: The mean state is the observation when available, 0 otherwise.
            return NumPyGaussianBelief(
                mean=np.linalg.pinv(H) @ observation,
                covariance=np.repeat(
                    initial_state_covariance[..., np.newaxis], K, axis=2
                ),
            )

        def blend(
            partitioning: ObstaclePartitioning,
            *,
            initial: NumPyGaussianBelief,
            prediction: NumPyGaussianBelief,
            update: NumPyGaussianBelief,
        ) -> NumPyGaussianBelief:
            return NumPyGaussianBelief(
                mean=np.where(
                    partitioning.should_update,
                    update.mean,
                    np.where(
                        partitioning.should_initialize, initial.mean, prediction.mean
                    ),
                ),
                covariance=np.where(
                    partitioning.should_update,
                    update.covariance,
                    np.where(
                        partitioning.should_initialize,
                        initial.covariance,
                        prediction.covariance,
                    ),
                ),
            )

        partitioning = partition(observation=observation, mean=prediction.mean)

        if np.all(partitioning.should_update):
            return update(
                observation=observation,
                mean=prediction.mean,
                covariance=prediction.covariance,
            )

        observation, mean, covariance = substitute_missing_values(
            observation=observation,
            mean=prediction.mean,
            covariance=prediction.covariance,
        )

        updated = update(observation=observation, mean=mean, covariance=covariance)

        if np.any(partitioning.should_initialize):
            initial = initialize_from(observation)
        else:
            initial = prediction

        return blend(
            partitioning, initial=initial, prediction=prediction, update=updated
        )

    @staticmethod
    def initial_belief_from(
        observations: Float[Array, "T D_z K"],
        *,
        initial_state_covariance: Float[Array, "D_x D_x"],
    ) -> NumPyGaussianBelief:
        D_x = initial_state_covariance.shape[0]
        K = observations.shape[2]

        mean = np.full((D_x, K), np.nan)
        covariance = np.full((D_x, D_x, K), np.nan)

        return NumPyGaussianBelief(mean=mean, covariance=covariance)

    @staticmethod
    def standardize_noise_covariance(
        covariance: NumPyNoiseCovarianceDescription, *, dimension: int
    ) -> Float[Array, "D_c D_c"]:
        if isinstance(covariance, (int, float)):
            return covariance * np.eye(dimension)

        match covariance.ndim:
            case 2:
                return covariance
            case 1:
                return np.diag(covariance)
            case _:
                assert False, f"Invalid covariance shape: {covariance.shape}"
