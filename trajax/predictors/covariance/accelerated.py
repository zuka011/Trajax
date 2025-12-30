from dataclasses import dataclass

from trajax.types import (
    JaxInitialPositionCovariance,
    JaxInitialVelocityCovariance,
)
from trajax.obstacles import JaxObstacleStates

import jax.numpy as jnp


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
