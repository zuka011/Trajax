from typing import NamedTuple
from dataclasses import dataclass

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


@dataclass(kw_only=True)
class NumPyKalmanFilter:
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
            observations,
            initial_state_covariance=initial_state_covariance,
            observation_matrix=observation_matrix,
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
    ) -> NumPyGaussianBelief[D_x, K]:
        """Performs the update step of the Kalman filter using a new observation.

        Args:
            observation: The newly observed state.
            prediction: The predicted belief from the prediction step.
            observation_matrix: H matrix mapping state to observation space.
            observation_noise_covariance: Q matrix representing the covariance of observation noise.
        """
        return numpy_kalman_filter.update(
            observation,
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

        This method provides a simple way to initialize the state mean based on the first
        observation, while using the provided initial state covariance for uncertainty.

        Args:
            observations: The observed state history up to the current time step.
            initial_state_covariance: Sigma_0 matrix representing initial state uncertainty.
            observation_matrix: H matrix mapping state to observation space.
        """
        return numpy_kalman_filter.initial_belief_from(
            observations,
            initial_state_covariance=initial_state_covariance,
            observation_matrix=observation_matrix,
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
    ) -> NumPyGaussianBelief[D_x, K]:
        H = observation_matrix
        Q = observation_noise_covariance
        mu, sigma = prediction
        D_x = mu.shape[0]

        S = np.einsum("ij,jlk,ml->imk", H, sigma, H) + Q[..., np.newaxis]
        rhs = np.einsum("ijk,lj->ilk", sigma, H)

        K = np.linalg.solve(S.transpose(2, 0, 1), rhs.transpose(2, 1, 0)).transpose(
            2, 1, 0
        )

        innovation = observation - H @ mu
        KH = np.einsum("ijk,jl->ilk", K, H)

        return NumPyGaussianBelief(
            mean=mu + np.einsum("ijk,jk->ik", K, innovation),
            covariance=np.einsum(
                "ijk,jlk->ilk", np.eye(D_x)[..., np.newaxis] - KH, sigma
            ),
        )

    @staticmethod
    def initial_belief_from[T: int, D_x: int, D_z: int, K: int](
        observations: Array[Dims[T, D_z, K]],
        *,
        initial_state_covariance: Array[Dims[D_x, D_x]],
        observation_matrix: Array[Dims[D_z, D_x]],
    ) -> NumPyGaussianBelief[D_x, K]:
        H_pinv = np.linalg.pinv(observation_matrix)
        K = observations.shape[2]

        return NumPyGaussianBelief(
            mean=H_pinv @ observations[0],
            covariance=np.repeat(initial_state_covariance[..., np.newaxis], K, axis=2),
        )

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
