from typing import Any
from dataclasses import dataclass

from trajax.types import (
    ObstacleStateSequences,
    JaxInitialPositionCovariance,
    JaxInitialVelocityCovariance,
)

import jax.numpy as jnp


@dataclass(kw_only=True, frozen=True)
class JaxConstantVarianceProvider:
    position_variance: float
    velocity_variance: float

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
