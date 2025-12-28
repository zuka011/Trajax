from dataclasses import dataclass

from trajax.types import CostFunction, Costs, CostSumFunction


@dataclass(frozen=True)
class CombinedCost[InputT, StateT, CostsT: Costs]:
    costs: list[CostFunction[InputT, StateT, CostsT]]
    sum: CostSumFunction[CostsT]

    def __post_init__(self) -> None:
        assert len(self.costs) > 0, "At least one cost function must be provided."

    def __call__(self, *, inputs: InputT, states: StateT) -> CostsT:
        costs = [
            cost_function(inputs=inputs, states=states) for cost_function in self.costs
        ]

        return self.sum(costs=costs, initial=costs[0].zero())
