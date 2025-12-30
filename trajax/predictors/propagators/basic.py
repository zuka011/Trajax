from dataclasses import dataclass

from trajax.types import (
    ObstacleStateSequences,
    NumPyInitialCovarianceProvider,
    NumPyPositionCovariance,
)

import numpy as np


@dataclass(kw_only=True, frozen=True)
class NumPyLinearCovariancePropagator[StateSequencesT: ObstacleStateSequences]:
    time_step: float
    initial_covariance: NumPyInitialCovarianceProvider[StateSequencesT]

    @staticmethod
    def create[S: ObstacleStateSequences](
        *, time_step_size: float, initial_covariance: NumPyInitialCovarianceProvider[S]
    ) -> "NumPyLinearCovariancePropagator[S]":
        return NumPyLinearCovariancePropagator(
            time_step=time_step_size, initial_covariance=initial_covariance
        )

    def propagate(self, *, states: StateSequencesT) -> NumPyPositionCovariance:
        position_cov = self.initial_covariance.position(states=states)
        velocity_cov = self.initial_covariance.velocity(states=states)

        t = np.arange(1, states.horizon + 1)[:, np.newaxis, np.newaxis, np.newaxis]

        return position_cov + (t * self.time_step**2) * velocity_cov
