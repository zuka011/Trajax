from dataclasses import dataclass

from trajax import types

from jaxtyping import Array as JaxArray, Float

import numpy as np
import jax.numpy as jnp


type NumPyObstacleStates[T: int, K: int] = types.numpy.ObstacleStates[T, K]
type NumPyInitialPositionCovariance[T: int, K: int] = (
    types.numpy.InitialPositionCovariance[K]
)
type NumPyInitialVelocityVariance[K: int] = types.numpy.InitialVelocityCovariance[K]
type NumPyInitialCovarianceProvider = types.numpy.InitialCovarianceProvider[
    NumPyObstacleStates
]

type JaxObstacleStates[T: int, K: int] = types.jax.ObstacleStates[T, K]
type JaxInitialPositionCovariance[K: int] = types.jax.InitialPositionCovariance[K]
type JaxInitialVelocityCovariance[K: int] = types.jax.InitialVelocityCovariance[K]
type JaxInitialCovarianceProvider = types.jax.InitialCovarianceProvider[
    JaxObstacleStates
]


@dataclass(kw_only=True, frozen=True)
class NumPyConstantVarianceProvider:
    position_variance: float
    velocity_variance: float

    def position[K: int](
        self, states: NumPyObstacleStates[int, K]
    ) -> NumPyInitialPositionCovariance[K]:
        return np.tile(
            (np.eye(2) * self.position_variance)[..., np.newaxis], (1, 1, states.count)
        )

    def velocity[K: int](
        self, states: NumPyObstacleStates[int, K]
    ) -> NumPyInitialVelocityVariance[K]:
        return np.tile(
            (np.eye(2) * self.velocity_variance)[..., np.newaxis], (1, 1, states.count)
        )


@dataclass(kw_only=True, frozen=True)
class NumPyConstantCovarianceProvider[K: int]:
    position_covariance: NumPyInitialPositionCovariance[K]
    velocity_covariance: NumPyInitialVelocityVariance[K]

    def position(
        self, states: NumPyObstacleStates[int, K]
    ) -> NumPyInitialPositionCovariance[K]:
        assert states.count == self.position_covariance.shape[2]
        return self.position_covariance

    def velocity(
        self, states: NumPyObstacleStates[int, K]
    ) -> NumPyInitialVelocityVariance[K]:
        assert states.count == self.velocity_covariance.shape[2]
        return self.velocity_covariance


@dataclass(kw_only=True, frozen=True)
class JaxConstantVarianceProvider:
    position_variance: float
    velocity_variance: float

    def position[K: int](
        self, states: JaxObstacleStates[int, K]
    ) -> JaxInitialPositionCovariance[K]:
        return jnp.tile(
            (jnp.eye(2) * self.position_variance)[..., jnp.newaxis],
            (1, 1, states.count),
        )

    def velocity[K: int](
        self, states: JaxObstacleStates[int, K]
    ) -> JaxInitialVelocityCovariance[K]:
        return jnp.tile(
            (jnp.eye(2) * self.velocity_variance)[..., jnp.newaxis],
            (1, 1, states.count),
        )


@dataclass(kw_only=True, frozen=True)
class JaxConstantCovarianceProvider[K: int]:
    position_covariance: Float[JaxArray, "2 2 K"]
    velocity_covariance: Float[JaxArray, "2 2 K"]

    def position(
        self, states: JaxObstacleStates[int, K]
    ) -> JaxInitialPositionCovariance[K]:
        assert states.count == self.position_covariance.shape[2]
        return self.position_covariance

    def velocity(
        self, states: JaxObstacleStates[int, K]
    ) -> JaxInitialVelocityCovariance[K]:
        assert states.count == self.velocity_covariance.shape[2]
        return self.velocity_covariance
