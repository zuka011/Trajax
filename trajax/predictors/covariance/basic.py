from dataclasses import dataclass

from trajax.types import (
    NumPyInitialPositionCovariance,
    NumPyInitialVelocityCovariance,
)
from trajax.obstacles import NumPyObstacleStates

import numpy as np


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
    ) -> NumPyInitialVelocityCovariance[K]:
        return np.tile(
            (np.eye(2) * self.velocity_variance)[..., np.newaxis], (1, 1, states.count)
        )
