from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    ObstacleStateSequences,
    JaxInitialPositionCovariance,
    JaxInitialVelocityCovariance,
    JaxInitialCovarianceProvider,
    JaxPositionCovariance,
)

from jaxtyping import Scalar

import jax
import jax.numpy as jnp


@dataclass(kw_only=True, frozen=True)
class JaxLinearCovariancePropagator[StateSequencesT: ObstacleStateSequences]:
    time_step: Scalar
    initial_covariance: JaxInitialCovarianceProvider[StateSequencesT]

    @staticmethod
    def create[S: ObstacleStateSequences](
        *, time_step_size: float, initial_covariance: JaxInitialCovarianceProvider[S]
    ) -> "JaxLinearCovariancePropagator[S]":
        return JaxLinearCovariancePropagator(
            time_step=jnp.asarray(time_step_size), initial_covariance=initial_covariance
        )

    def propagate(self, *, states: StateSequencesT) -> JaxPositionCovariance:
        return propagate(
            position_covariance=self.initial_covariance.position(states=states),
            velocity_covariance=self.initial_covariance.velocity(states=states),
            time_step=self.time_step,
            horizon=states.horizon,
        )


@jax.jit(static_argnames=("horizon",))
@jaxtyped
def propagate(
    *,
    position_covariance: JaxInitialPositionCovariance,
    velocity_covariance: JaxInitialVelocityCovariance,
    time_step: Scalar,
    horizon: int,
) -> JaxPositionCovariance:
    t = jnp.arange(1, horizon + 1)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    return position_covariance + (t * time_step**2) * velocity_covariance
