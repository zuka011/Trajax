from typing import Final

from trajax.costs.basic import (
    ContouringCost as NumPyContouringCost,
    LagCost as NumPyLagCost,
    ProgressCost as NumPyProgressCost,
    ControlSmoothingCost as NumPyControlSmoothingCost,
)
from trajax.costs.accelerated import (
    ContouringCost as JaxContouringCost,
    LagCost as JaxLagCost,
    ProgressCost as JaxProgressCost,
    ControlSmoothingCost as JaxControlSmoothingCost,
)
from trajax.costs.combined import CombinedCost, NumPyCostSumFunction
from trajax.mppi.common import ControlInputBatch, StateBatch, CostFunction
from trajax.mppi.basic import Costs as NumPyCosts


class costs:
    class numpy:
        @staticmethod
        def combined[
            ControlInputBatchT: ControlInputBatch,
            StateBatchT: StateBatch,
            CostsT: NumPyCosts,
        ](
            *costs: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
        ) -> CombinedCost[ControlInputBatchT, StateBatchT, CostsT]:
            """Creates a NumPy cost function combining all given cost functions by summation."""
            return CombinedCost(costs=list(costs), sum=NumPyCostSumFunction())

        class tracking:
            contouring: Final = NumPyContouringCost.create
            lag: Final = NumPyLagCost.create
            progress: Final = NumPyProgressCost.create

        class comfort:
            control_smoothing: Final = NumPyControlSmoothingCost.create

    class jax:
        class tracking:
            contouring: Final = JaxContouringCost.create
            lag: Final = JaxLagCost.create
            progress: Final = JaxProgressCost.create

        class comfort:
            control_smoothing: Final = JaxControlSmoothingCost.create
