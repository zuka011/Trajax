from dataclasses import dataclass

from trajax.types import (
    ObstacleStateSequences,
    NumPyInitialCovarianceProvider,
    NumPyPositionCovariance,
)
from trajax.predictors.common import CovariancePadding

from numtypes import Dims, Array

import numpy as np


type NumPyPaddedPositionCovariance[T: int, D_c: int, K: int] = Array[
    Dims[T, D_c, D_c, K]
]


@dataclass(kw_only=True, frozen=True)
class NumPyLinearCovariancePropagator[StateSequencesT: ObstacleStateSequences]:
    time_step: float
    initial_covariance: NumPyInitialCovarianceProvider[StateSequencesT]
    padding: CovariancePadding

    @staticmethod
    def create[S: ObstacleStateSequences](
        *,
        time_step_size: float,
        initial_covariance: NumPyInitialCovarianceProvider[S],
        padding: CovariancePadding = CovariancePadding.create(
            to_dimension=2, epsilon=1e9
        ),
    ) -> "NumPyLinearCovariancePropagator[S]":
        return NumPyLinearCovariancePropagator(
            time_step=time_step_size,
            initial_covariance=initial_covariance,
            padding=padding,
        )

    def propagate(self, *, states: StateSequencesT) -> NumPyPaddedPositionCovariance:
        position_cov = self.initial_covariance.position(states=states)
        velocity_cov = self.initial_covariance.velocity(states=states)

        t = np.arange(1, states.horizon + 1)[:, np.newaxis, np.newaxis, np.newaxis]

        return self.pad(position_cov + (t * self.time_step**2) * velocity_cov)

    def pad(
        self, covariances: NumPyPositionCovariance
    ) -> NumPyPaddedPositionCovariance:
        if (dimension := covariances.shape[1]) == self.padding.to_dimension:
            return covariances

        assert dimension < self.padding.to_dimension, (
            f"Covariance dimension is {dimension}, "
            f"which exceeds the target dimension {self.padding.to_dimension}. You need to specify "
            f"the final dimension the covariance should be padded to (i.e. >= {dimension})."
        )

        pad_amount = self.padding.to_dimension - dimension
        padded = np.pad(
            covariances,
            ((0, 0), (0, pad_amount), (0, pad_amount), (0, 0)),
            constant_values=0,
        )

        for i in range(dimension, self.padding.to_dimension):
            padded[:, i, i, :] = self.padding.epsilon

        return padded
