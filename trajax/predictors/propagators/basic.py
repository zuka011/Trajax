from typing import Any, cast
from dataclasses import dataclass

from trajax.types import (
    ObstacleModel,
    ObstacleStateSequences,
    CovariancePropagator,
    NumPyCovarianceProvider,
)
from trajax.predictors.common import CovarianceResizing

from numtypes import Dims, Array

import numpy as np


type NumPyCovarianceArray[T: int = int, D_o: int = int, K: int = int] = Array[
    Dims[T, D_o, D_o, K]
]
type NumPyCovarianceResizing[CovarianceSequencesT] = CovarianceResizing[
    NumPyCovarianceArray, CovarianceSequencesT
]


@dataclass(kw_only=True, frozen=True)
class NumPyLinearCovariancePropagator[
    StateSequencesT: ObstacleStateSequences,
    CovarianceSequencesT: NumPyCovarianceArray,
](CovariancePropagator[StateSequencesT, CovarianceSequencesT]):
    """NumPy linear covariance propagator for constant-velocity motion.

    Implements the formula: P(t) = P_0 + t² * Δt² * Σ_u
    where P_0 is initial state covariance and Σ_u is input covariance.

    This is a special case of EKF where F = I and G = Δt * I, which reduces
    the general formula P_{t+1} = F P_t F^T + G Σ_u G^T to P_{t+1} = P_t + Δt² * Σ_u.
    """

    time_step: float
    covariance: NumPyCovarianceProvider
    resizing: NumPyCovarianceResizing[CovarianceSequencesT]

    @staticmethod
    def create[S: ObstacleStateSequences, C: NumPyCovarianceArray](
        *,
        time_step_size: float,
        covariance: NumPyCovarianceProvider[S, C, C],
        resizing: NumPyCovarianceResizing[C] = CovarianceResizing.create(
            pad_to=2, epsilon=1e-9
        ),
    ) -> "NumPyLinearCovariancePropagator[S, C]":
        return NumPyLinearCovariancePropagator(
            time_step=time_step_size,
            covariance=covariance,
            resizing=resizing,
        )

    def propagate(
        self, *, states: StateSequencesT, inputs: Any = None
    ) -> CovarianceSequencesT:
        P_0 = self.covariance.state(states)
        Sigma_u = self.covariance.input(states)

        t = np.arange(1, states.horizon + 1)[:, np.newaxis, np.newaxis, np.newaxis]

        return apply_resizing(
            P_0 + (t * self.time_step**2) * Sigma_u,
            self.resizing,
        )


@dataclass(kw_only=True, frozen=True)
class NumPyEkfCovariancePropagator[
    StateSequencesT: ObstacleStateSequences,
    CovarianceSequencesT: NumPyCovarianceArray,
](CovariancePropagator[StateSequencesT, CovarianceSequencesT]):
    """NumPy EKF covariance propagator using state-dependent Jacobian linearization.

    Implements the formula: P_{t+1} = F_t P_t F_t^T + G_t Σ_u G_t^T
    where:
    - P_0 is the initial state covariance
    - Σ_u is the input covariance (uncertainty about control inputs)
    - F_t = ∂f/∂x is the state Jacobian
    - G_t = ∂f/∂u is the input Jacobian
    """

    model: ObstacleModel
    covariance: NumPyCovarianceProvider
    resizing: NumPyCovarianceResizing[CovarianceSequencesT]

    @staticmethod
    def create[S: ObstacleStateSequences, C: NumPyCovarianceArray](
        *,
        model: ObstacleModel,
        covariance: NumPyCovarianceProvider[S, C, C],
        resizing: NumPyCovarianceResizing[C] = CovarianceResizing.identity(),
    ) -> "NumPyEkfCovariancePropagator[S, C]":
        return NumPyEkfCovariancePropagator(
            model=model,
            covariance=covariance,
            resizing=resizing,
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

        return cast(
            CovarianceSequencesT,
            propagate_ekf(
                P_0,
                input_covariance=Sigma_u,
                state_jacobians=F,
                input_jacobians=G,
                horizon=states.horizon,
                count=states.count,
                resizing=self.resizing,
            ),
        )


def propagate_ekf(
    initial_P: Array,
    *,
    input_covariance: Array,
    state_jacobians: Array,
    input_jacobians: Array,
    horizon: int,
    count: int,
    resizing: CovarianceResizing,
) -> NumPyCovarianceArray:
    """Propagates covariance using P_{t+1} = F P_t F^T + G Σ_u G^T."""
    state_dimension = initial_P.shape[0]
    covariances = np.zeros((horizon, state_dimension, state_dimension, count))

    P = initial_P
    for t in range(horizon):
        F = state_jacobians[t]
        G = input_jacobians[t]

        # P_{t+1} = F P_t F^T + G Σ_u G^T
        # Using einsum: 'ijk,jlk,mlk->imk' for F @ P @ F.T
        F_P_Ft = np.einsum("ijk,jlk,mlk->imk", F, P, F)
        # Using einsum: 'ijk,jlk,mlk->imk' for G @ Σ_u @ G.T
        G_Sigma_Gt = np.einsum("ijk,jlk,mlk->imk", G, input_covariance, G)

        P = F_P_Ft + G_Sigma_Gt
        # Symmetrize to prevent floating-point drift
        P = (P + P.swapaxes(0, 1)) / 2
        covariances[t] = P

    return apply_resizing(covariances, resizing)


def apply_resizing[CovarianceSequencesT: NumPyCovarianceArray](
    covariances: NumPyCovarianceArray,
    resizing: NumPyCovarianceResizing[CovarianceSequencesT],
) -> CovarianceSequencesT:

    extracted = resizing.keep(covariances)

    if resizing.pad_to is None or (dimension := extracted.shape[1]) == resizing.pad_to:
        return extracted

    assert dimension < resizing.pad_to, (
        f"Covariance dimension is {dimension} (after extraction), "
        f"which exceeds the target dimension {resizing.pad_to}. You need to specify "
        f"the final dimension the covariance should be padded to (i.e. >= {dimension})."
    )

    pad_amount = resizing.pad_to - dimension
    padded = np.pad(
        extracted,
        ((0, 0), (0, pad_amount), (0, pad_amount), (0, 0)),
        constant_values=0,
    )

    for i in range(dimension, resizing.pad_to):
        padded[:, i, i, :] = resizing.epsilon

    return cast(CovarianceSequencesT, padded)
