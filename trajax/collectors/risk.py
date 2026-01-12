from typing import Sequence
from dataclasses import dataclass, field

from trajax.types import Risk, RiskMetric
from trajax.collectors.access import access


@dataclass(frozen=True)
class RiskCollector[
    CostFunctionT,
    StateBatchT,
    ObstacleStatesT,
    SamplerT,
    RiskT = Risk,
](RiskMetric[CostFunctionT, StateBatchT, ObstacleStatesT, SamplerT, RiskT]):
    inner: RiskMetric[CostFunctionT, StateBatchT, ObstacleStatesT, SamplerT, RiskT]
    _collected: list[RiskT] = field(default_factory=list)

    @staticmethod
    def decorating[CF, SB, OS, S, R](
        metric: RiskMetric[CF, SB, OS, S, R],
    ) -> "RiskCollector[CF, SB, OS, S, R]":
        return RiskCollector(metric)

    def compute(
        self,
        cost_function: CostFunctionT,
        *,
        states: StateBatchT,
        obstacle_states: ObstacleStatesT,
        sampler: SamplerT,
    ) -> RiskT:
        self._collected.append(
            risk := self.inner.compute(
                cost_function,
                states=states,
                obstacle_states=obstacle_states,
                sampler=sampler,
            )
        )

        return risk

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def collected(self) -> Sequence[RiskT]:
        return self._collected

    @property
    def key(self) -> str:
        return access.risks.key
