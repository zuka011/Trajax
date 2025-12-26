from typing import Protocol
from dataclasses import dataclass

from trajax.mppi import StateBatch
from trajax.costs.collision import (
    NumPyObstacleStates,
    NumPySampledObstacleStates,
    NumPyObstacleStateSampler,
    SampleCostFunction,
)
from trajax.costs.risk.base import StateTrajectories, ObstacleStateUncertainties

from numtypes import Array, Dims

import riskit as rk


class NumPyRiskMetric(Protocol):
    def compute[
        StateT: StateBatch,
        ObstacleStateT: NumPyObstacleStates,
        SampledObstacleStateT: NumPySampledObstacleStates,
        T: int,
        M: int,
    ](
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Array[Dims[T, M, int]]
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: NumPyObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> Array[Dims[T, M]]:
        """Computes the risk metric based on the provided cost function and returns it as a NumPy array."""
        ...


@dataclass(kw_only=True, frozen=True)
class NumPyMeanVarianceMetric:
    gamma: float
    sampler: rk.Sampler

    @staticmethod
    def create(*, gamma: float, sample_count: int) -> "NumPyMeanVarianceMetric":
        return NumPyMeanVarianceMetric(
            gamma=gamma, sampler=rk.sampler.monte_carlo(sample_count)
        )

    def compute[
        StateT: StateBatch,
        ObstacleStateT: NumPyObstacleStates,
        SampledObstacleStateT: NumPySampledObstacleStates,
        T: int,
        M: int,
        N: int,
    ](
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Array[Dims[T, M, N]]
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: NumPyObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> Array[Dims[T, M]]:
        def J(
            *, trajectories: StateT, uncertainties: SampledObstacleStateT
        ) -> rk.NumPyCosts[T, M, N]:
            return cost_function(states=trajectories, samples=uncertainties)

        metric = rk.risk.mean_variance_of(J, gamma=self.gamma).sampled_with(
            self.sampler
        )

        return metric.compute(
            trajectories=StateTrajectories(states),
            uncertainties=ObstacleStateUncertainties(
                obstacle_states=obstacle_states, sampler=sampler
            ),
        )
