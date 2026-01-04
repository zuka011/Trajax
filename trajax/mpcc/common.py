from typing import NamedTuple


class MpccMppiSetup[MppiT, ModelT, ContouringCostT, LagCostT](NamedTuple):
    mppi: MppiT
    model: ModelT
    contouring_cost: ContouringCostT
    lag_cost: LagCostT
