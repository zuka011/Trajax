from typing import Any
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    ObstacleStateSequences,
)

from numtypes import Array
from jaxtyping import Array as JaxArray, Float, Scalar

import jax.numpy as jnp


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxConstantVarianceProvider:
    """Provides constant isotropic covariance matrices.

    Returns a covariance matrix of shape (dimension, dimension, K) where
    K is the number of obstacles. The covariance is isotropic (diagonal)
    with the specified variance on all diagonal elements.
    """

    variance: Scalar
    dimension: int

    @staticmethod
    def create(
        *,
        variance: float | Scalar,
        dimension: int,
    ) -> "JaxConstantVarianceProvider":
        return JaxConstantVarianceProvider(
            variance=jnp.asarray(variance),
            dimension=dimension,
        )

    def __call__[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> Float[JaxArray, "D D K"]:
        return jnp.tile(
            (jnp.eye(self.dimension) * self.variance)[..., jnp.newaxis],
            (1, 1, states.count),
        )


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxConstantCovarianceProvider:
    """Provides a fixed covariance matrix.

    Returns the same covariance matrix for all obstacles. The provided matrix
    should have shape (dimension, dimension, K) where K matches the obstacle count.
    """

    covariance: Float[JaxArray, "D D K"]

    @staticmethod
    def create(
        *,
        covariance: Array | Float[JaxArray, "D D K"],
    ) -> "JaxConstantCovarianceProvider":
        return JaxConstantCovarianceProvider(covariance=jnp.asarray(covariance))

    def __call__[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> Float[JaxArray, "D D K"]:
        assert states.count == self.covariance.shape[2], (
            f"Covariance shape {self.covariance.shape} does not match "
            f"obstacle count {states.count}."
        )
        return self.covariance


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxFullStateVarianceProvider:
    """Provides full-state isotropic covariance matrices.

    Uses the state dimension from the obstacle states to build a full covariance
    matrix. Useful when you want the covariance to match the state dimension.
    """

    variance: Scalar

    @staticmethod
    def create(*, variance: float | Scalar) -> "JaxFullStateVarianceProvider":
        return JaxFullStateVarianceProvider(variance=jnp.asarray(variance))

    def __call__[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> Float[JaxArray, "D_o D_o K"]:
        return jnp.tile(
            (jnp.eye(states.dimension) * self.variance)[..., jnp.newaxis],
            (1, 1, states.count),
        )


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxZeroProcessNoiseProvider:
    """Provides zero process noise (for when process noise should be disabled)."""

    dimension: int

    @staticmethod
    def create(*, dimension: int) -> "JaxZeroProcessNoiseProvider":
        return JaxZeroProcessNoiseProvider(dimension=dimension)

    def __call__[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> Float[JaxArray, "D D K"]:
        return jnp.zeros((self.dimension, self.dimension, states.count))


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxFullStateZeroProcessNoiseProvider:
    """Provides zero process noise that matches the full state dimension."""

    @staticmethod
    def create() -> "JaxFullStateZeroProcessNoiseProvider":
        return JaxFullStateZeroProcessNoiseProvider()

    def __call__[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> Float[JaxArray, "D_o D_o K"]:
        return jnp.zeros((states.dimension, states.dimension, states.count))


# --- Unified Covariance Providers ---


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxCovarianceProviderComposite:
    """Composite covariance provider that combines state and input providers.

    Implements the CovarianceProvider protocol with separate state() and input()
    methods. This allows using any existing provider implementation for state
    covariance and a separate one for input covariance.
    """

    state_provider: Any  # Callable[[ObstacleStateSequences], JaxArray]
    input_provider: Any  # Callable[[ObstacleStateSequences], JaxArray]

    @staticmethod
    def create(
        *,
        state_provider: Any,
        input_provider: Any,
    ) -> "JaxCovarianceProviderComposite":
        return JaxCovarianceProviderComposite(
            state_provider=state_provider,
            input_provider=input_provider,
        )

    def state[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> Float[JaxArray, "D D K"]:
        return self.state_provider(states)

    def input[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> Float[JaxArray, "D D K"]:
        return self.input_provider(states)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxIsotropicCovarianceProvider:
    """Provides isotropic (diagonal) covariance matrices for state and input.

    This is a convenient unified provider for the common case where both
    state and input covariances are isotropic with specified variances.
    """

    state_variance: Scalar
    state_dimension: int
    input_variance: Scalar
    input_dimension: int

    @staticmethod
    def create(
        *,
        state_variance: float | Scalar,
        state_dimension: int,
        input_variance: float | Scalar,
        input_dimension: int,
    ) -> "JaxIsotropicCovarianceProvider":
        return JaxIsotropicCovarianceProvider(
            state_variance=jnp.asarray(state_variance),
            state_dimension=state_dimension,
            input_variance=jnp.asarray(input_variance),
            input_dimension=input_dimension,
        )

    def state[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> Float[JaxArray, "D_o D_o K"]:
        return jnp.tile(
            (jnp.eye(self.state_dimension) * self.state_variance)[..., jnp.newaxis],
            (1, 1, states.count),
        )

    def input[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> Float[JaxArray, "D_u D_u K"]:
        return jnp.tile(
            (jnp.eye(self.input_dimension) * self.input_variance)[..., jnp.newaxis],
            (1, 1, states.count),
        )
