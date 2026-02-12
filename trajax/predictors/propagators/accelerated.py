from typing import Any, NamedTuple
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    ObstacleModel,
    ObstacleStateSequences,
    CovariancePropagator,
    CovarianceExtractor,
    JaxCovarianceProvider,
)
from trajax.predictors.common import CovarianceResizing

from jaxtyping import Float, Array as JaxArray, Scalar

import jax
import jax.numpy as jnp


type JaxCovarianceArray[T: int, D_o: int, K: int] = Float[JaxArray, "T D_o D_o K"]


class JaxCovarianceResizing[CovarianceSequencesT](NamedTuple):
    keep: CovarianceExtractor[JaxCovarianceArray, CovarianceSequencesT]
    pad_to: int | None
    epsilon: Scalar

    @staticmethod
    def of[C](
        resizing: CovarianceResizing[JaxCovarianceArray, C],
    ) -> "JaxCovarianceResizing[C]":
        return JaxCovarianceResizing(
            keep=resizing.keep,
            pad_to=resizing.pad_to,
            epsilon=jnp.asarray(resizing.epsilon),
        )


@dataclass(kw_only=True, frozen=True)
class JaxLinearCovariancePropagator[
    StateSequencesT: ObstacleStateSequences,
    CovarianceSequencesT: JaxCovarianceArray,
](CovariancePropagator[StateSequencesT, CovarianceSequencesT]):
    """JAX linear covariance propagator for constant-velocity motion.

    Implements the formula: P(t) = P_0 + t² * Δt² * Σ_u
    where P_0 is initial state covariance and Σ_u is input covariance.

    This is a special case of EKF where F = I and G = Δt * I, which reduces
    the general formula P_{t+1} = F P_t F^T + G Σ_u G^T to P_{t+1} = P_t + Δt² * Σ_u.
    """

    time_step: Scalar
    covariance: JaxCovarianceProvider
    resizing: JaxCovarianceResizing[CovarianceSequencesT]

    @staticmethod
    def create[S: ObstacleStateSequences, C: JaxCovarianceArray](
        *,
        time_step_size: float,
        covariance: JaxCovarianceProvider[S, C, C],
        resizing: CovarianceResizing[JaxCovarianceArray, C] = CovarianceResizing.create(
            pad_to=2, epsilon=1e-9
        ),
    ) -> "JaxLinearCovariancePropagator[S, C]":

        return JaxLinearCovariancePropagator(
            time_step=jnp.asarray(time_step_size),
            covariance=covariance,
            resizing=JaxCovarianceResizing.of(resizing),
        )

    def propagate(
        self, *, states: StateSequencesT, inputs: Any = None
    ) -> CovarianceSequencesT:
        return propagate_linear(
            initial_covariance=self.covariance.state(states),
            input_covariance=self.covariance.input(states),
            time_step=self.time_step,
            horizon=states.horizon,
            keep=self.resizing.keep,
            pad_to=self.resizing.pad_to,
            epsilon=self.resizing.epsilon,
        )


@dataclass(kw_only=True, frozen=True)
class JaxEkfCovariancePropagator[
    StateSequencesT: ObstacleStateSequences,
    CovarianceSequencesT: JaxCovarianceArray,
](CovariancePropagator[StateSequencesT, CovarianceSequencesT]):
    """JAX EKF covariance propagator using state-dependent Jacobian linearization.

    Implements the formula: P_{t+1} = F_t P_t F_t^T + G_t Σ_u G_t^T
    where:
    - P_0 is the initial state covariance
    - Σ_u is the input covariance (uncertainty about control inputs)
    - F_t = ∂f/∂x is the state Jacobian
    - G_t = ∂f/∂u is the input Jacobian
    """

    model: ObstacleModel
    covariance: JaxCovarianceProvider
    resizing: JaxCovarianceResizing[CovarianceSequencesT]

    @staticmethod
    def create[S: ObstacleStateSequences, C: JaxCovarianceArray](
        *,
        model: ObstacleModel,
        covariance: JaxCovarianceProvider[S, C, C],
        resizing: CovarianceResizing[
            JaxCovarianceArray, C
        ] = CovarianceResizing.identity(),
    ) -> "JaxEkfCovariancePropagator[S, C]":
        return JaxEkfCovariancePropagator(
            model=model,
            covariance=covariance,
            resizing=JaxCovarianceResizing.of(resizing),
        )

    def propagate(
        self, *, states: StateSequencesT, inputs: Any = None
    ) -> CovarianceSequencesT:
        assert inputs is not None, (
            "EKF propagator requires control inputs to compute state-dependent Jacobians."
        )

        P_0 = self.covariance.state(states)
        Sigma_u = self.covariance.input(states)
        F = self.model.state_jacobian(states=states, inputs=inputs)
        G = self.model.input_jacobian(states=states, inputs=inputs)

        return propagate_ekf(
            initial_covariance=P_0,
            input_covariance=Sigma_u,
            state_jacobians=F,
            input_jacobians=G,
            keep=self.resizing.keep,
            pad_to=self.resizing.pad_to,
            epsilon=self.resizing.epsilon,
        )


@jax.jit(static_argnames=("horizon", "keep", "pad_to"))
@jaxtyped
def propagate_linear(
    *,
    initial_covariance: Float[JaxArray, "D_o D_o K"],
    input_covariance: Float[JaxArray, "D_o D_o K"],
    time_step: Scalar,
    horizon: int,
    keep: CovarianceExtractor,
    pad_to: int | None,
    epsilon: Scalar,
) -> Float[JaxArray, "T D_p D_p K"]:
    t = jnp.arange(1, horizon + 1)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    return apply_resizing(
        initial_covariance + (t * time_step**2) * input_covariance,
        keep=keep,
        pad_to=pad_to,
        epsilon=epsilon,
    )


@jax.jit(static_argnames=("keep", "pad_to"))
def propagate_ekf(
    *,
    initial_covariance: Float[JaxArray, "D_o D_o K"],
    input_covariance: Float[JaxArray, "D_u D_u K"],
    state_jacobians: Float[JaxArray, "T D_o D_o K"],
    input_jacobians: Float[JaxArray, "T D_o D_u K"],
    keep: CovarianceExtractor,
    pad_to: int | None,
    epsilon: Scalar,
) -> JaxCovarianceArray:
    """Propagates covariance using P_{t+1} = F P_t F^T + G Σ_u G^T."""

    def ekf_step(
        P: Float[JaxArray, "D_o D_o K"],
        jacobians: tuple[Float[JaxArray, "D_o D_o K"], Float[JaxArray, "D_o D_u K"]],
    ) -> tuple[Float[JaxArray, "D_o D_o K"], Float[JaxArray, "D_o D_o K"]]:
        F, G = jacobians

        # P_{t+1} = F P_t F^T + G Σ_u G^T
        F_P_Ft = jnp.einsum("ijk,jlk,mlk->imk", F, P, F)
        G_Sigma_Gt = jnp.einsum("ijk,jlk,mlk->imk", G, input_covariance, G)

        P_next = F_P_Ft + G_Sigma_Gt
        # Symmetrize to prevent floating-point drift
        P_next = (P_next + P_next.swapaxes(0, 1)) / 2
        return P_next, P_next

    _, covariances = jax.lax.scan(
        ekf_step, initial_covariance, (state_jacobians, input_jacobians)
    )

    return apply_resizing(
        covariances,
        keep=keep,
        pad_to=pad_to,
        epsilon=epsilon,
    )


@jax.jit(static_argnames=("keep", "pad_to"))
@jaxtyped
def apply_resizing(
    covariances: Float[JaxArray, "T D_o D_o K"],
    *,
    keep: CovarianceExtractor,
    pad_to: int | None,
    epsilon: Scalar,
) -> Float[JaxArray, "T D_p D_p K"]:
    extracted = keep(covariances)

    if pad_to is None or (dimension := extracted.shape[1]) == pad_to:
        return extracted

    pad_amount = pad_to - dimension
    padded = jnp.pad(
        extracted,
        ((0, 0), (0, pad_amount), (0, pad_amount), (0, 0)),
        constant_values=0,
    )

    for i in range(pad_amount):
        padded = padded.at[:, dimension + i, dimension + i, :].set(epsilon)

    return padded
