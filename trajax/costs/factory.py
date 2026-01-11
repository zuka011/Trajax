from typing import Final

from trajax.types import (
    ControlInputBatch,
    StateBatch,
    CostFunction,
    NumPyCosts,
    JaxCosts,
)
from trajax.costs.basic import (
    NumPyContouringCost,
    NumPyLagCost,
    NumPyProgressCost,
    NumPyControlSmoothingCost,
)
from trajax.costs.accelerated import (
    JaxContouringCost,
    JaxLagCost,
    JaxProgressCost,
    JaxControlSmoothingCost,
)
from trajax.costs.combined import CombinedCost, NumPyCostSumFunction, JaxCostSumFunction
from trajax.costs.collision import NumPyCollisionCost, JaxCollisionCost
from trajax.costs.distance import (
    NumPyCircleDistanceExtractor,
    JaxCircleDistanceExtractor,
)
from trajax.costs.boundary import (
    NumPyBoundaryCost,
    NumPyFixedWidthBoundary,
    JaxBoundaryCost,
    JaxFixedWidthBoundary,
)


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

        class safety:
            collision: Final = NumPyCollisionCost.create
            boundary: Final = NumPyBoundaryCost.create

    class jax:
        @staticmethod
        def combined[
            ControlInputBatchT: ControlInputBatch,
            StateBatchT: StateBatch,
            CostsT: JaxCosts,
        ](
            *costs: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
        ) -> CombinedCost[ControlInputBatchT, StateBatchT, CostsT]:
            """Creates a JAX cost function combining all given cost functions by summation."""
            return CombinedCost(costs=list(costs), sum=JaxCostSumFunction())

        class tracking:
            contouring: Final = JaxContouringCost.create
            lag: Final = JaxLagCost.create
            progress: Final = JaxProgressCost.create

        class comfort:
            control_smoothing: Final = JaxControlSmoothingCost.create

        class safety:
            collision: Final = JaxCollisionCost.create
            boundary: Final = JaxBoundaryCost.create


class distance:
    class numpy:
        circles: Final = NumPyCircleDistanceExtractor.create

    class jax:
        circles: Final = JaxCircleDistanceExtractor.create


class boundary:
    class numpy:
        fixed_width: Final = NumPyFixedWidthBoundary.create

    class jax:
        fixed_width: Final = JaxFixedWidthBoundary.create
