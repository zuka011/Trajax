from typing import Any
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    ObstacleStateSequences,
    JaxInitialPositionCovariance,
    JaxInitialVelocityCovariance,
)

from numtypes import Array, Dims
from jaxtyping import Array as JaxArray, Float, Scalar

import jax.numpy as jnp


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxConstantVarianceProvider:
    position_variance: Scalar
    velocity_variance: Scalar

    @staticmethod
    def create(
        *,
        position_variance: float | Scalar,
        velocity_variance: float | Scalar,
    ) -> "JaxConstantVarianceProvider":
        return JaxConstantVarianceProvider(
            position_variance=jnp.asarray(position_variance),
            velocity_variance=jnp.asarray(velocity_variance),
        )

    def position[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> JaxInitialPositionCovariance[K]:
        return jnp.tile(
            (jnp.eye(2) * self.position_variance)[..., jnp.newaxis],
            (1, 1, states.count),
        )

    def velocity[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> JaxInitialVelocityCovariance[K]:
        return jnp.tile(
            (jnp.eye(2) * self.velocity_variance)[..., jnp.newaxis],
            (1, 1, states.count),
        )


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxConstantCovarianceProvider[K: int]:
    position_covariance: Float[JaxArray, "2 2 K"]
    velocity_covariance: Float[JaxArray, "2 2 K"]

    @staticmethod
    def create(
        *,
        position_covariance: Array[Dims[K]] | Float[JaxArray, "2 2 K"],
        velocity_covariance: Array[Dims[K]] | Float[JaxArray, "2 2 K"],
    ) -> "JaxConstantCovarianceProvider[K]":
        return JaxConstantCovarianceProvider(
            position_covariance=jnp.asarray(position_covariance),
            velocity_covariance=jnp.asarray(velocity_covariance),
        )

    def position(
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> JaxInitialPositionCovariance[K]:
        assert states.count == self.position_covariance.shape[2]
        return self.position_covariance

    def velocity(
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> JaxInitialVelocityCovariance[K]:
        assert states.count == self.velocity_covariance.shape[2]
        return self.velocity_covariance
