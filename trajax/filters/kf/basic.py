from typing import NamedTuple

from numtypes import Dims, Array, shape_of

import numpy as np


type NumPyNoiseCovarianceArrayDescription[D_c: int] = (
    Array[Dims[D_c, D_c]] | Array[Dims[D_c]]
)

type NumPyNoiseCovarianceDescription[D_c: int] = (
    NumPyNoiseCovarianceArrayDescription[D_c] | float
)


class NumPyGaussianBelief[D_x: int, K: int](NamedTuple):
    mean: Array[Dims[D_x, K]]
    covariance: Array[Dims[D_x, D_x, K]]


class ObstaclePartitioning[K: int](NamedTuple):
    should_update: Array[Dims[K]]
    should_initialize: Array[Dims[K]]


class NumPyKalmanFilter:
    """Kalman Filter for linear systems."""

    @staticmethod
    def create() -> "NumPyKalmanFilter":
        return NumPyKalmanFilter()

    def filter[T: int, D_x: int, D_z: int, K: int](
        self,
        observations: Array[Dims[T, D_z, K]],
        *,
        initial_state_covariance: Array[Dims[D_x, D_x]],
        state_transition_matrix: Array[Dims[D_x, D_x]],
        process_noise_covariance: Array[Dims[D_x, D_x]],
        observation_noise_covariance: Array[Dims[D_z, D_z]],
        observation_matrix: Array[Dims[D_z, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
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

        for observation in observations:
            belief = self.predict(
                belief=belief,
                state_transition_matrix=state_transition_matrix,
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
        state_transition_matrix: Array[Dims[D_x, D_x]],
        process_noise_covariance: Array[Dims[D_x, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
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

    def update[D_x: int, D_z: int, K: int](
        self,
        observation: Array[Dims[D_z, K]],
        *,
        prediction: NumPyGaussianBelief[D_x, K],
        observation_matrix: Array[Dims[D_z, D_x]],
        observation_noise_covariance: Array[Dims[D_z, D_z]],
        initial_state_covariance: Array[Dims[D_x, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
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

    def initial_belief_from[T: int, D_x: int, D_z: int, K: int](
        self,
        observations: Array[Dims[T, D_z, K]],
        *,
        initial_state_covariance: Array[Dims[D_x, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
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
    def predict[D_x: int, K: int](
        *,
        belief: NumPyGaussianBelief[D_x, K],
        state_transition_matrix: Array[Dims[D_x, D_x]],
        process_noise_covariance: Array[Dims[D_x, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
        A = state_transition_matrix
        R = process_noise_covariance
        mu, sigma = belief

        return NumPyGaussianBelief(
            mean=A @ mu,
            covariance=np.einsum("ij,jlk,ml->imk", A, sigma, A) + R[..., np.newaxis],
        )

    @staticmethod
    def update[D_x: int, D_z: int, K: int](
        observation: Array[Dims[D_z, K]],
        *,
        prediction: NumPyGaussianBelief[D_x, K],
        observation_matrix: Array[Dims[D_z, D_x]],
        observation_noise_covariance: Array[Dims[D_z, D_z]],
        initial_state_covariance: Array[Dims[D_x, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
        H = observation_matrix
        Q = observation_noise_covariance

        def partition(
            *, observation: Array[Dims[D_z, K]], mean: Array[Dims[D_x, K]]
        ) -> ObstaclePartitioning[K]:
            observation_valid = ~np.any(np.isnan(observation), axis=0)
            prediction_valid = ~np.any(np.isnan(mean), axis=0)

            return ObstaclePartitioning(
                should_update=observation_valid & prediction_valid,
                should_initialize=observation_valid & ~prediction_valid,
            )

        def substitute_missing_values(
            *,
            observation: Array[Dims[D_z, K]],
            mean: Array[Dims[D_x, K]],
            covariance: Array[Dims[D_x, D_x, K]],
        ) -> tuple[Array[Dims[D_z, K]], Array[Dims[D_x, K]], Array[Dims[D_x, D_x, K]]]:
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
            observation: Array[Dims[D_z, K]],
            mean: Array[Dims[D_x, K]],
            covariance: Array[Dims[D_x, D_x, K]],
        ) -> NumPyGaussianBelief[D_x, K]:
            D_x = prediction.mean.shape[0]

            S = np.einsum("ij,jlk,ml->imk", H, covariance, H) + Q[..., np.newaxis]
            rhs = np.einsum("ijk,lj->ilk", covariance, H)
            K = np.linalg.solve(S.transpose(2, 0, 1), rhs.transpose(2, 1, 0)).transpose(
                2, 1, 0
            )

            innovation = observation - H @ mean
            KH = np.einsum("ijk,jl->ilk", K, H)

            return NumPyGaussianBelief(
                mean=mean + np.einsum("ijk,jk->ik", K, innovation),
                covariance=np.einsum(
                    "ijk,jlk->ilk", np.eye(D_x)[..., np.newaxis] - KH, covariance
                ),
            )

        def initialize_from(
            observation: Array[Dims[D_z, K]],
        ) -> NumPyGaussianBelief[D_x, K]:
            K = observation.shape[1]

            # NOTE: The mean state is the observation when available, 0 otherwise.
            return NumPyGaussianBelief(
                mean=np.linalg.pinv(H) @ observation,
                covariance=np.repeat(
                    initial_state_covariance[..., np.newaxis], K, axis=2
                ),
            )

        def blend(
            partitioning: ObstaclePartitioning[K],
            *,
            initial: NumPyGaussianBelief[D_x, K],
            prediction: NumPyGaussianBelief[D_x, K],
            update: NumPyGaussianBelief[D_x, K],
        ) -> NumPyGaussianBelief[D_x, K]:
            return NumPyGaussianBelief(
                mean=np.select(  # type: ignore
                    [partitioning.should_update, partitioning.should_initialize],  # type: ignore
                    [update.mean, initial.mean],
                    default=prediction.mean,
                ),
                covariance=np.select(  # type: ignore
                    [partitioning.should_update, partitioning.should_initialize],  # type: ignore
                    [update.covariance, initial.covariance],
                    default=prediction.covariance,
                ),
            )

        partitioning = partition(observation=observation, mean=prediction.mean)
        observation, mean, covariance = substitute_missing_values(
            observation=observation,
            mean=prediction.mean,
            covariance=prediction.covariance,
        )

        return blend(
            partitioning,
            initial=initialize_from(observation),
            prediction=prediction,
            update=update(observation=observation, mean=mean, covariance=covariance),
        )

    @staticmethod
    def initial_belief_from[T: int, D_x: int, D_z: int, K: int](
        observations: Array[Dims[T, D_z, K]],
        *,
        initial_state_covariance: Array[Dims[D_x, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
        D_x = initial_state_covariance.shape[0]
        K = observations.shape[2]

        mean = np.full((D_x, K), np.nan)
        covariance = np.full((D_x, D_x, K), np.nan)

        assert shape_of(mean, matches=(D_x, K), name="initial mean")
        assert shape_of(covariance, matches=(D_x, D_x, K), name="initial covariance")

        return NumPyGaussianBelief(mean=mean, covariance=covariance)

    @staticmethod
    def standardize_noise_covariance[D_c: int](
        covariance: NumPyNoiseCovarianceDescription[D_c], *, dimension: D_c
    ) -> Array[Dims[D_c, D_c]]:
        if isinstance(covariance, (int, float)):
            result = covariance * np.eye(dimension)
        else:
            match covariance.ndim:
                case 2:
                    result = covariance
                case 1:
                    result = np.diag(covariance)
                case _:
                    assert False, f"Invalid covariance shape: {covariance.shape}"

        assert shape_of(result, matches=(dimension, dimension), name="covariance")

        return result
