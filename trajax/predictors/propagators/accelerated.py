from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    ObstacleStateSequences,
    JaxInitialPositionCovariance,
    JaxInitialVelocityCovariance,
    JaxInitialCovarianceProvider,
    JaxPositionCovariance,
)
from trajax.predictors.common import CovariancePadding

from jaxtyping import Float, Array as JaxArray, Scalar

import jax
import jax.numpy as jnp


type JaxPaddedPositionCovariance[T: int, D: int, K: int] = Float[JaxArray, "T D D K"]


@dataclass(kw_only=True, frozen=True)
class JaxLinearCovariancePropagator[StateSequencesT: ObstacleStateSequences]:
    time_step: Scalar
    epsilon: Scalar
    to_dimension: int
    initial_covariance: JaxInitialCovarianceProvider[StateSequencesT]

    @staticmethod
    def create[S: ObstacleStateSequences](
        *,
        time_step_size: float,
        initial_covariance: JaxInitialCovarianceProvider[S],
        padding: CovariancePadding = CovariancePadding.create(
            to_dimension=2, epsilon=1e9
        ),
    ) -> "JaxLinearCovariancePropagator[S]":
        assert padding.to_dimension >= 2, (
            f"Padding target dimension must be at least 2, got {padding.to_dimension}. "
            f"You need to specify the final dimension the covariance should be padded to (i.e. >= 2), "
            "not the amount to pad by."
        )

        return JaxLinearCovariancePropagator(
            time_step=jnp.asarray(time_step_size),
            epsilon=jnp.asarray(padding.epsilon),
            to_dimension=padding.to_dimension,
            initial_covariance=initial_covariance,
        )

    def propagate(self, *, states: StateSequencesT) -> JaxPaddedPositionCovariance:
        return propagate(
            position_covariance=self.initial_covariance.position(states=states),
            velocity_covariance=self.initial_covariance.velocity(states=states),
            time_step=self.time_step,
            horizon=states.horizon,
            to_dimension=self.to_dimension,
            epsilon=self.epsilon,
        )


@jax.jit(static_argnames=("horizon", "to_dimension"))
@jaxtyped
def propagate(
    *,
    position_covariance: JaxInitialPositionCovariance,
    velocity_covariance: JaxInitialVelocityCovariance,
    time_step: Scalar,
    epsilon: Scalar,
    horizon: int,
    to_dimension: int,
) -> JaxPaddedPositionCovariance:
    t = jnp.arange(1, horizon + 1)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    return pad(
        position_covariance + (t * time_step**2) * velocity_covariance,
        to_dimension=to_dimension,
        epsilon=epsilon,
    )


@jax.jit(static_argnames=("to_dimension",))
@jaxtyped
def pad(
    covariances: JaxPositionCovariance,
    *,
    to_dimension: int,
    epsilon: Scalar,
) -> JaxPaddedPositionCovariance:
    dimension = covariances.shape[1]
    pad_amount = to_dimension - dimension

    padded = jnp.pad(
        covariances,
        ((0, 0), (0, pad_amount), (0, pad_amount), (0, 0)),
        constant_values=0,
    )

    for i in range(pad_amount):
        padded = padded.at[:, dimension + i, dimension + i, :].set(epsilon)

    return padded
