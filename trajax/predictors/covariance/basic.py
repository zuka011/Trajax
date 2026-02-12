from typing import Any
from dataclasses import dataclass

from trajax.types import ObstacleStateSequences

from numtypes import Array

import numpy as np


@dataclass(kw_only=True, frozen=True)
class NumPyConstantVarianceProvider:
    """Provides constant isotropic covariance matrices.

    Returns a covariance matrix of shape (dimension, dimension, K) where
    K is the number of obstacles. The covariance is isotropic (diagonal)
    with the specified variance on all diagonal elements.
    """

    variance: float
    dimension: int

    @staticmethod
    def create(
        *,
        variance: float,
        dimension: int,
    ) -> "NumPyConstantVarianceProvider":
        return NumPyConstantVarianceProvider(variance=variance, dimension=dimension)

    def __call__[K: int](self, states: ObstacleStateSequences[int, K, Any]) -> Array:
        return np.tile(
            (np.eye(self.dimension) * self.variance)[..., np.newaxis],
            (1, 1, states.count),
        )


@dataclass(kw_only=True, frozen=True)
class NumPyConstantCovarianceProvider:
    """Provides a fixed covariance matrix.

    Returns the same covariance matrix for all obstacles. The provided matrix
    should have shape (dimension, dimension, K) where K matches the obstacle count.
    """

    covariance: Array  # Shape: (D, D, K)

    @staticmethod
    def create(*, covariance: Array) -> "NumPyConstantCovarianceProvider":
        return NumPyConstantCovarianceProvider(covariance=covariance)

    def __call__[K: int](self, states: ObstacleStateSequences[int, K, Any]) -> Array:
        assert states.count == self.covariance.shape[2], (
            f"Covariance shape {self.covariance.shape} does not match "
            f"obstacle count {states.count}."
        )
        return self.covariance


@dataclass(kw_only=True, frozen=True)
class NumPyFullStateVarianceProvider:
    """Provides full-state isotropic covariance matrices.

    Uses the state dimension from the obstacle states to build a full covariance
    matrix. Useful when you want the covariance to match the state dimension.
    """

    variance: float

    @staticmethod
    def create(*, variance: float) -> "NumPyFullStateVarianceProvider":
        return NumPyFullStateVarianceProvider(variance=variance)

    def __call__[K: int](self, states: ObstacleStateSequences[int, K, Any]) -> Array:
        return np.tile(
            (np.eye(states.dimension) * self.variance)[..., np.newaxis],
            (1, 1, states.count),
        )


@dataclass(kw_only=True, frozen=True)
class NumPyZeroProcessNoiseProvider:
    """Provides zero process noise (for when process noise should be disabled)."""

    dimension: int

    @staticmethod
    def create(*, dimension: int) -> "NumPyZeroProcessNoiseProvider":
        return NumPyZeroProcessNoiseProvider(dimension=dimension)

    def __call__[K: int](self, states: ObstacleStateSequences[int, K, Any]) -> Array:
        return np.zeros((self.dimension, self.dimension, states.count))


@dataclass(kw_only=True, frozen=True)
class NumPyFullStateZeroProcessNoiseProvider:
    """Provides zero process noise that matches the full state dimension."""

    @staticmethod
    def create() -> "NumPyFullStateZeroProcessNoiseProvider":
        return NumPyFullStateZeroProcessNoiseProvider()

    def __call__[K: int](self, states: ObstacleStateSequences[int, K, Any]) -> Array:
        return np.zeros((states.dimension, states.dimension, states.count))


# --- Unified Covariance Providers ---


@dataclass(kw_only=True, frozen=True)
class NumPyCovarianceProviderComposite:
    """Composite covariance provider that combines state and input providers.

    Implements the CovarianceProvider protocol with separate state() and input()
    methods. This allows using any existing provider implementation for state
    covariance and a separate one for input covariance.
    """

    state_provider: Any  # Callable[[ObstacleStateSequences], Array]
    input_provider: Any  # Callable[[ObstacleStateSequences], Array]

    @staticmethod
    def create(
        *,
        state_provider: Any,
        input_provider: Any,
    ) -> "NumPyCovarianceProviderComposite":
        return NumPyCovarianceProviderComposite(
            state_provider=state_provider,
            input_provider=input_provider,
        )

    def state[K: int](self, states: ObstacleStateSequences[int, K, Any]) -> Array:
        return self.state_provider(states)

    def input[K: int](self, states: ObstacleStateSequences[int, K, Any]) -> Array:
        return self.input_provider(states)


@dataclass(kw_only=True, frozen=True)
class NumPyIsotropicCovarianceProvider:
    """Provides isotropic (diagonal) covariance matrices for state and input.

    This is a convenient unified provider for the common case where both
    state and input covariances are isotropic with specified variances.
    """

    state_variance: float
    state_dimension: int
    input_variance: float
    input_dimension: int

    @staticmethod
    def create(
        *,
        state_variance: float,
        state_dimension: int,
        input_variance: float,
        input_dimension: int,
    ) -> "NumPyIsotropicCovarianceProvider":
        return NumPyIsotropicCovarianceProvider(
            state_variance=state_variance,
            state_dimension=state_dimension,
            input_variance=input_variance,
            input_dimension=input_dimension,
        )

    def state[K: int](self, states: ObstacleStateSequences[int, K, Any]) -> Array:
        return np.tile(
            (np.eye(self.state_dimension) * self.state_variance)[..., np.newaxis],
            (1, 1, states.count),
        )

    def input[K: int](self, states: ObstacleStateSequences[int, K, Any]) -> Array:
        return np.tile(
            (np.eye(self.input_dimension) * self.input_variance)[..., np.newaxis],
            (1, 1, states.count),
        )
