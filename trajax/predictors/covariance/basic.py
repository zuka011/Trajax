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

    @staticmethod
    def create(
        *,
        position_variance: float,
        velocity_variance: float,
    ) -> "NumPyConstantVarianceProvider":
        return NumPyConstantVarianceProvider(
            position_variance=position_variance, velocity_variance=velocity_variance
        )

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


@dataclass(kw_only=True, frozen=True)
class NumPyConstantCovarianceProvider[K: int]:
    position_covariance: NumPyInitialPositionCovariance[K]
    velocity_covariance: NumPyInitialVelocityCovariance[K]

    @staticmethod
    def create(
        *,
        position_covariance: NumPyInitialPositionCovariance[K],
        velocity_covariance: NumPyInitialVelocityCovariance[K],
    ) -> "NumPyConstantCovarianceProvider[K]":
        return NumPyConstantCovarianceProvider(
            position_covariance=position_covariance,
            velocity_covariance=velocity_covariance,
        )

    def position(
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> NumPyInitialPositionCovariance[K]:
        assert states.count == self.position_covariance.shape[2]
        return self.position_covariance

    def velocity(
        self, states: ObstacleStateSequences[int, K, Any]
    ) -> NumPyInitialVelocityCovariance[K]:
        assert states.count == self.velocity_covariance.shape[2]
        return self.velocity_covariance
