from typing import Protocol, Sequence
from dataclasses import dataclass

from trajax.mppi import (
    StateBatch,
    ControlInputBatch,
    CostFunction,
    Costs,
)


class CostSumFunction[CostsT: Costs](Protocol):
    def __call__(self, costs: Sequence[CostsT], *, initial: CostsT) -> CostsT:
        """Sums an iterable of costs into a single cost."""
        ...


@dataclass(frozen=True)
class CombinedCost[InputT: ControlInputBatch, StateT: StateBatch, CostsT: Costs]:
    costs: list[CostFunction[InputT, StateT, CostsT]]
    sum: CostSumFunction[CostsT]

    def __post_init__(self) -> None:
        assert len(self.costs) > 0, "At least one cost function must be provided."

    def __call__(self, *, inputs: InputT, states: StateT) -> CostsT:
        costs = [
            cost_function(inputs=inputs, states=states) for cost_function in self.costs
        ]

        return self.sum(costs=costs, initial=costs[0].zero())
