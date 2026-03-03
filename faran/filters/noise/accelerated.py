from typing import NamedTuple

from faran.types import (
    jaxtyped,
    JaxNoiseModel,
    JaxNoiseModelProvider,
    JaxGaussianBelief,
    JaxNoiseCovariances,
)

from jaxtyping import Array as JaxArray, Float, Int

import jax
import jax.numpy as jnp


class JaxClampedNoiseModel[StateT](NamedTuple):
    """Decorator that clamps an inner noise model's output diagonals to a floor."""

    inner: JaxNoiseModel[StateT]
    floor: JaxNoiseCovariances

    @property
    def state(self) -> StateT:
        return self.inner.state

    def __call__(
        self,
        *,
        noise: JaxNoiseCovariances,
        prediction: JaxGaussianBelief,
        observation: Float[JaxArray, "D_z K"],
        state: StateT,
    ) -> tuple[JaxNoiseCovariances, StateT]:
        result, state = self.inner(
            noise=noise, prediction=prediction, observation=observation, state=state
        )
        return JaxNoiseCovariances(
            process_noise_covariance=apply_diagonal_floor(
                result.process_noise_covariance,
                floor=self.floor.process_noise_covariance,
            ),
            observation_noise_covariance=apply_diagonal_floor(
                result.observation_noise_covariance,
                floor=self.floor.observation_noise_covariance,
            ),
        ), state


class JaxClampedNoiseProvider[StateT](NamedTuple):
    floor: JaxNoiseCovariances
    inner: JaxNoiseModelProvider[StateT]

    @staticmethod
    def decorate[S](
        inner: JaxNoiseModelProvider[S], *, floor: JaxNoiseCovariances
    ) -> "JaxClampedNoiseProvider[S]":
        """Creates a noise model provider that clamps the diagonal of the
        noise covariances to the specified floor.

        Args:
            inner: The inner noise model provider to delegate to.
            floor: Minimum noise covariances. Diagonal entries of the inner model's
                output will be clamped to be no smaller than these.
        """
        return JaxClampedNoiseProvider(inner=inner, floor=floor)

    def __call__(
        self,
        *,
        observation_matrix: Float[JaxArray, "D_z D_x"],
        noise: JaxNoiseCovariances,
    ) -> JaxClampedNoiseModel:
        inner_model = self.inner(observation_matrix=observation_matrix, noise=noise)
        return JaxClampedNoiseModel(inner=inner_model, floor=self.floor)


class JaxAdaptiveNoiseState(NamedTuple):
    """Circular buffer state for adaptive noise estimation."""

    buffer: Float[JaxArray, "W D_z"]
    entry_count: Int[JaxArray, ""]


class JaxAdaptiveNoise(NamedTuple):
    """Innovation-Based Adaptive Estimation (IAE) for noise covariances."""

    observation_matrix: Float[JaxArray, "D_z D_x"]
    window_size: int

    @property
    def state(self) -> JaxAdaptiveNoiseState:
        observation_dimension = self.observation_matrix.shape[0]
        return JaxAdaptiveNoiseState(
            buffer=jnp.zeros((self.window_size, observation_dimension)),
            entry_count=jnp.array(0, dtype=jnp.int32),
        )

    @jax.jit
    @jaxtyped
    def __call__(
        self,
        *,
        noise: JaxNoiseCovariances,
        prediction: JaxGaussianBelief,
        observation: Float[JaxArray, "D_z K"],
        state: JaxAdaptiveNoiseState,
    ) -> tuple[JaxNoiseCovariances, JaxAdaptiveNoiseState]:
        has_nan = jnp.any(jnp.isnan(prediction.mean)) | jnp.any(
            jnp.isnan(prediction.covariance)
        )

        return jax.lax.cond(
            ~has_nan,
            lambda _: self.adapt(
                noise=noise, prediction=prediction, observation=observation, state=state
            ),
            lambda _: (noise, state),
            None,
        )

    @jax.jit
    @jaxtyped
    def adapt(
        self,
        *,
        noise: JaxNoiseCovariances,
        prediction: JaxGaussianBelief,
        observation: Float[JaxArray, "D_z K"],
        state: JaxAdaptiveNoiseState,
    ) -> tuple[JaxNoiseCovariances, JaxAdaptiveNoiseState]:
        innovation = observation - self.observation_matrix @ prediction.mean
        mean_innovation = jnp.mean(innovation, axis=1)

        index = state.entry_count % self.window_size
        new_buffer = state.buffer.at[index].set(mean_innovation)
        new_count = state.entry_count + 1
        new_state = JaxAdaptiveNoiseState(buffer=new_buffer, entry_count=new_count)

        buffer_full = new_count >= self.window_size

        def return_original(
            _: None,
        ) -> tuple[JaxNoiseCovariances, JaxAdaptiveNoiseState]:
            return noise, new_state

        def compute_adapted(
            _: None,
        ) -> tuple[JaxNoiseCovariances, JaxAdaptiveNoiseState]:
            innovation_matrix = compute_innovation_matrix(new_buffer)
            mean_covariance = jnp.mean(prediction.covariance, axis=2)
            kalman_gain = compute_kalman_gain(
                mean_covariance=mean_covariance,
                observation_matrix=self.observation_matrix,
                observation_noise_covariance=noise.observation_noise_covariance,
            )

            adapted_noise = JaxNoiseCovariances(
                process_noise_covariance=enforce_spd(
                    kalman_gain @ innovation_matrix @ kalman_gain.T
                ),
                observation_noise_covariance=enforce_spd(
                    innovation_matrix
                    - self.observation_matrix
                    @ mean_covariance
                    @ self.observation_matrix.T
                ),
            )
            return adapted_noise, new_state

        return jax.lax.cond(buffer_full, compute_adapted, return_original, None)


class JaxAdaptiveNoiseProvider(NamedTuple):
    window_size: int = 20

    @staticmethod
    def create(*, window_size: int = 20) -> "JaxAdaptiveNoiseProvider":
        """Creates an innovation-based adaptive estimation model for noise.

        Args:
            window_size: The number of past observations considered, when adapting
                the noise covariances.
        """
        return JaxAdaptiveNoiseProvider(window_size=window_size)

    def __call__(
        self,
        *,
        observation_matrix: Float[JaxArray, "D_z D_x"],
        noise: JaxNoiseCovariances,
    ) -> JaxAdaptiveNoise:
        return JaxAdaptiveNoise(
            observation_matrix=observation_matrix, window_size=self.window_size
        )


@jax.jit
@jaxtyped
def compute_innovation_matrix(
    buffer: Float[JaxArray, "W D_z"],
) -> Float[JaxArray, "D_z D_z"]:
    return (buffer.T @ buffer) / buffer.shape[0]


@jax.jit
@jaxtyped
def compute_kalman_gain(
    *,
    mean_covariance: Float[JaxArray, "D_x D_x"],
    observation_matrix: Float[JaxArray, "D_z D_x"],
    observation_noise_covariance: Float[JaxArray, "D_z D_z"],
) -> Float[JaxArray, "D_x D_z"]:
    S = (
        observation_matrix @ mean_covariance @ observation_matrix.T
        + observation_noise_covariance
    )
    return mean_covariance @ observation_matrix.T @ jnp.linalg.inv(S)


@jax.jit
@jaxtyped
def apply_diagonal_floor(
    matrix: Float[JaxArray, "N N"], *, floor: Float[JaxArray, "N N"]
) -> Float[JaxArray, "N N"]:
    floored = jnp.maximum(jnp.diag(matrix), jnp.diag(floor))
    return matrix - jnp.diag(jnp.diag(matrix)) + jnp.diag(floored)


@jax.jit
@jaxtyped
def enforce_spd(matrix: Float[JaxArray, "N N"]) -> Float[JaxArray, "N N"]:
    symmetrised = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = jnp.linalg.eigh(symmetrised)
    return eigenvectors @ jnp.diag(jnp.maximum(eigenvalues, 0.0)) @ eigenvectors.T
