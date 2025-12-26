from dataclasses import dataclass

from trajax.mppi import StateBatch
from trajax.costs.collision import (
    JaxObstacleStates,
    JaxSampledObstacleStates,
    JaxObstacleStateSampler,
    SampleCostFunction,
)
from trajax.costs.risk.base import StateTrajectories, ObstacleStateUncertainties

from jaxtyping import Array as JaxArray, Float

import riskit as rk


@dataclass(kw_only=True, frozen=True)
class JaxMeanVarianceMetric:
    gamma: float
    sampler: rk.Sampler

    @staticmethod
    def create(*, gamma: float, sample_count: int) -> "JaxMeanVarianceMetric":
        return JaxMeanVarianceMetric(
            gamma=gamma, sampler=rk.sampler.monte_carlo(sample_count)
        )

    def compute[
        StateT: StateBatch,
        ObstacleStateT: JaxObstacleStates,
        SampledObstacleStateT: JaxSampledObstacleStates,
    ](
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Float[JaxArray, "T M N"]
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: JaxObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> Float[JaxArray, "T M"]:
        def J(
            *, trajectories: StateT, uncertainties: SampledObstacleStateT
        ) -> rk.JaxCosts:
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
