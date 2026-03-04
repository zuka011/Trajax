from typing import NamedTuple

from faran.types import (
    Array,
    NumPyNoiseModel,
    NumPyNoiseModelProvider,
    NumPyGaussianBelief,
    NumPyNoiseCovariances,
)

from jaxtyping import Float

import numpy as np


class NumPyClampedNoiseModel[StateT](NamedTuple):
    """Decorator that clamps an inner noise model's output diagonals to a floor."""

    inner: NumPyNoiseModel
    floor: NumPyNoiseCovariances

    def __call__(
        self,
        *,
        noise: NumPyNoiseCovariances,
        prediction: NumPyGaussianBelief,
        observation: Float[Array, "D_z K"],
        state: StateT,
    ) -> tuple[NumPyNoiseCovariances, StateT]:
        result, state = self.inner(
            noise=noise, prediction=prediction, observation=observation, state=state
        )
        return NumPyNoiseCovariances(
            process_noise_covariance=apply_diagonal_floor(
                result.process_noise_covariance,
                floor=self.floor.process_noise_covariance,
            ),
            observation_noise_covariance=apply_diagonal_floor(
                result.observation_noise_covariance,
                floor=self.floor.observation_noise_covariance,
            ),
        ), state

    @property
    def state(self) -> StateT:
        return self.inner.state


class NumPyClampedNoiseProvider[StateT](NamedTuple):
    inner: NumPyNoiseModelProvider[StateT]
    floor: NumPyNoiseCovariances

    @staticmethod
    def decorate[S](
        inner: NumPyNoiseModelProvider[S], *, floor: NumPyNoiseCovariances
    ) -> "NumPyClampedNoiseProvider[S]":
        """Creates a noise model provider that clamps the diagonal of the
        noise covariances to the specified floor.

        Args:
            inner: The inner noise model provider to delegate to.
            floor: Minimum noise covariances. Diagonal entries of the inner model's
                output will be clamped to be no smaller than these.
        """
        return NumPyClampedNoiseProvider(floor=floor, inner=inner)

    def __call__(
        self,
        *,
        observation_matrix: Float[Array, "D_z D_x"],
        noise: NumPyNoiseCovariances,
    ) -> NumPyClampedNoiseModel:
        inner_model = self.inner(observation_matrix=observation_matrix, noise=noise)
        return NumPyClampedNoiseModel(inner=inner_model, floor=self.floor)


class NumPyAdaptiveNoiseState(NamedTuple):
    buffer: list[Float[Array, " D_z"]]


class NumPyAdaptiveNoise(NamedTuple):
    """Innovation-Based Adaptive Estimation (IAE) for noise covariances."""

    observation_matrix: Float[Array, "D_z D_x"]
    window_size: int

    def __call__(
        self,
        *,
        noise: NumPyNoiseCovariances,
        prediction: NumPyGaussianBelief,
        observation: Float[Array, "D_z K"],
        state: NumPyAdaptiveNoiseState,
    ) -> tuple[NumPyNoiseCovariances, NumPyAdaptiveNoiseState]:
        if np.any(np.isnan(prediction.mean)) or np.any(np.isnan(prediction.covariance)):
            return noise, state

        innovation = observation - self.observation_matrix @ prediction.mean
        state.buffer.append(np.mean(innovation, axis=1))

        if len(state.buffer) > self.window_size:
            state.buffer.pop(0)

        if len(state.buffer) < self.window_size:
            return noise, state

        innovation_matrix = compute_innovation_matrix(state.buffer)
        mean_covariance = np.mean(prediction.covariance, axis=2)
        kalman_gain = compute_kalman_gain(
            mean_covariance=mean_covariance,
            observation_matrix=self.observation_matrix,
            observation_noise_covariance=noise.observation_noise_covariance,
        )

        adapted_noise = NumPyNoiseCovariances(
            process_noise_covariance=enforce_spd(
                kalman_gain @ innovation_matrix @ kalman_gain.T
            ),
            observation_noise_covariance=enforce_spd(
                innovation_matrix
                - self.observation_matrix @ mean_covariance @ self.observation_matrix.T
            ),
        )

        return adapted_noise, state

    @property
    def state(self) -> NumPyAdaptiveNoiseState:
        return NumPyAdaptiveNoiseState(buffer=[])


class NumPyAdaptiveNoiseProvider(NamedTuple):
    window_size: int

    @staticmethod
    def create(*, window_size: int) -> "NumPyAdaptiveNoiseProvider":
        """Creates an innovation-based adaptive estimation model for noise.

        Args:
            window_size: The number of past observations considered, when adapting
                the noise covariances.
        """
        return NumPyAdaptiveNoiseProvider(window_size=window_size)

    def __call__(
        self,
        *,
        observation_matrix: Float[Array, "D_z D_x"],
        noise: NumPyNoiseCovariances,
    ) -> NumPyAdaptiveNoise:
        return NumPyAdaptiveNoise(
            observation_matrix=observation_matrix, window_size=self.window_size
        )


def compute_innovation_matrix(
    buffer: list[Float[Array, " D_z"]],
) -> Float[Array, "D_z D_z"]:
    innovations = np.stack(buffer)
    return (innovations.T @ innovations) / len(buffer)


def compute_kalman_gain(
    *,
    mean_covariance: Float[Array, "D_x D_x"],
    observation_matrix: Float[Array, "D_z D_x"],
    observation_noise_covariance: Float[Array, "D_z D_z"],
) -> Float[Array, "D_x D_z"]:
    S = (
        observation_matrix @ mean_covariance @ observation_matrix.T
        + observation_noise_covariance
    )
    return mean_covariance @ observation_matrix.T @ np.linalg.inv(S)


def apply_diagonal_floor(
    matrix: Float[Array, "N N"], *, floor: Float[Array, "N N"]
) -> Float[Array, "N N"]:
    floored = np.maximum(np.diag(matrix), np.diag(floor))
    return matrix - np.diag(np.diag(matrix)) + np.diag(floored)


def enforce_spd(matrix: Float[Array, "N N"]) -> Float[Array, "N N"]:
    symmetrised = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetrised)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
