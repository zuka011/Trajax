from typing import Protocol
from dataclasses import dataclass

from trajax.types import StateBatch, ObstacleStateSampler, SampleCostFunction

import riskit as rk


class RiskMetricCreator[
    StateT,
    SampledObstacleStateT,
    CostsT: rk.Costs,
    RiskT: rk.Risk,
    ArrayT: rk.ArrayLike,
](Protocol):
    def __call__(
        self,
        *,
        cost: rk.BatchCostFunction[StateT, SampledObstacleStateT, CostsT],
        backend: rk.Backend[CostsT, RiskT, ArrayT],
    ) -> rk.RiskMetric[StateT, SampledObstacleStateT, RiskT]:
        """Creates a risk metric based on the provided cost function and backend."""
        ...


class RiskConverter[RkRiskT: rk.Risk, RiskT](Protocol):
    def __call__(self, risk: RkRiskT, /) -> RiskT:
        """Converts the given risk from the backend's risk type to the desired risk type."""
        ...


@dataclass(kw_only=True, frozen=True)
class RisKitRiskMetric[
    StateT: StateBatch,
    SampleT,
    RiskT,
    RkRiskT: rk.Risk,
    CostsT: rk.Costs,
    ArrayT: rk.ArrayLike,
]:
    backend: rk.Backend[CostsT, RkRiskT, ArrayT]
    creator: RiskMetricCreator[StateT, SampleT, CostsT, RkRiskT, ArrayT]
    to_risk: RiskConverter[RkRiskT, RiskT]
    _name: str

    @staticmethod
    def create[SB: StateBatch, SOS, R, RKR: rk.Risk, C: rk.Costs, A: rk.ArrayLike](
        *,
        backend: rk.Backend[C, RKR, A],
        creator: RiskMetricCreator[SB, SOS, C, RKR, A],
        to_risk: RiskConverter[RKR, R],
        name: str,
    ) -> "RisKitRiskMetric[SB, SOS, R, RKR, C, A]":
        return RisKitRiskMetric(
            backend=backend, creator=creator, to_risk=to_risk, _name=name
        )

    def compute[ObstacleStateT](
        self,
        cost_function: SampleCostFunction[StateT, SampleT, CostsT],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: ObstacleStateSampler[ObstacleStateT, SampleT],
    ) -> RiskT:
        def cost(*, trajectories: StateT, uncertainties: SampleT) -> CostsT:
            return cost_function(states=trajectories, samples=uncertainties)

        return self.to_risk(
            self.creator(cost=cost, backend=self.backend).compute(
                trajectories=StateTrajectories(states),
                uncertainties=ObstacleStateUncertainties(
                    obstacle_states=obstacle_states, sampler=sampler
                ),
            )
        )

    @property
    def name(self) -> str:
        return self._name


@dataclass(frozen=True)
class StateTrajectories[StateT: StateBatch]:
    states: StateT

    def get(self) -> StateT:
        return self.states

    @property
    def time_steps(self) -> int:
        return self.states.horizon

    @property
    def trajectory_count(self) -> int:
        return self.states.rollout_count


@dataclass(frozen=True)
class ObstacleStateUncertainties[StateT, SampleT]:
    obstacle_states: StateT
    sampler: ObstacleStateSampler[StateT, SampleT]

    def sample(self, count: int) -> SampleT:
        return self.sampler(self.obstacle_states, count=count)
