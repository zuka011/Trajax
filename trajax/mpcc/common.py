from typing import NamedTuple


class MpccMppiSetup[MppiT, ModelT, ContouringCostT, LagCostT](NamedTuple):
    """Named tuple bundling an MPCC-configured MPPI planner with its model and cost components."""

    mppi: MppiT
    model: ModelT
    contouring_cost: ContouringCostT
    lag_cost: LagCostT
