from dataclasses import dataclass, field

from trajax.types import Risk, RiskMetric
from trajax.collectors.access import access
from trajax.collectors.registry import (
    DataTransformer,
    IdentityTransformer,
    ModificationNotifierMixin,
    ListCollectorMixin,
    OnModifyCallback,
)


@dataclass(frozen=True)
class RiskCollector[
    CostFunctionT,
    StateBatchT,
    ObstacleStatesT,
    SamplerT,
    RiskT = Risk,
](
    RiskMetric[CostFunctionT, StateBatchT, ObstacleStatesT, SamplerT, RiskT],
    ModificationNotifierMixin,
    ListCollectorMixin,
):
    inner: RiskMetric[CostFunctionT, StateBatchT, ObstacleStatesT, SamplerT, RiskT]
    transformer: DataTransformer[RiskT]
    _callbacks: list[OnModifyCallback] = field(default_factory=list)
    _collected: list[RiskT] = field(default_factory=list)

    @staticmethod
    def decorating[CF, SB, OS, S, R](
        metric: RiskMetric[CF, SB, OS, S, R],
        *,
        transformer: DataTransformer[R] = IdentityTransformer(),
    ) -> "RiskCollector[CF, SB, OS, S, R]":
        return RiskCollector(metric, transformer=transformer)

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

        # TODO: Missing notify call.

        return risk

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def key(self) -> str:
        return access.risks.key
