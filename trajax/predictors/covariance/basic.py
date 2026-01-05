from typing import Any
from dataclasses import dataclass

from trajax.types import (
    ObstacleStateSequences,
    NumPyInitialPositionCovariance,
    NumPyInitialVelocityCovariance,
)

import numpy as np


@dataclass(kw_only=True, frozen=True)
class NumPyConstantVarianceProvider:
    position_variance: float
    velocity_variance: float

    def position[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> NumPyInitialPositionCovariance[K]:
        return np.tile(
            (np.eye(2) * self.position_variance)[..., np.newaxis], (1, 1, states.count)
        )

    def velocity[K: int](
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> NumPyInitialVelocityCovariance[K]:
        return np.tile(
            (np.eye(2) * self.velocity_variance)[..., np.newaxis], (1, 1, states.count)
        )
