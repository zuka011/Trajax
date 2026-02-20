from typing import Final

from faran.types import (
    ControlInputBatch,
    StateBatch,
    CostFunction,
    NumPyCosts,
    JaxCosts,
)
from faran.costs.basic import (
    NumPyContouringCost,
    NumPyLagCost,
    NumPyProgressCost,
    NumPyControlSmoothingCost,
    NumPyControlEffortCost,
)
from faran.costs.accelerated import (
    JaxContouringCost,
    JaxLagCost,
    JaxProgressCost,
    JaxControlSmoothingCost,
    JaxControlEffortCost,
)
from faran.costs.combined import CombinedCost, NumPyCostSumFunction, JaxCostSumFunction
from faran.costs.collision import NumPyCollisionCost, JaxCollisionCost
from faran.costs.distance import (
    NumPyCircleDistanceExtractor,
    JaxCircleDistanceExtractor,
    NumPySatDistanceExtractor,
    JaxSatDistanceExtractor,
)
from faran.costs.boundary import (
    NumPyBoundaryCost,
    NumPyFixedWidthBoundary,
    NumPyPiecewiseFixedWidthBoundary,
    JaxBoundaryCost,
    JaxFixedWidthBoundary,
    JaxPiecewiseFixedWidthBoundary,
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
            control_effort: Final = NumPyControlEffortCost.create

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
            control_effort: Final = JaxControlEffortCost.create

        class safety:
            collision: Final = JaxCollisionCost.create
            boundary: Final = JaxBoundaryCost.create


class distance:
    class numpy:
        circles: Final = NumPyCircleDistanceExtractor.create
        sat: Final = NumPySatDistanceExtractor.create

    class jax:
        circles: Final = JaxCircleDistanceExtractor.create
        sat: Final = JaxSatDistanceExtractor.create


class boundary:
    class numpy:
        fixed_width: Final = NumPyFixedWidthBoundary.create
        piecewise_fixed_width: Final = NumPyPiecewiseFixedWidthBoundary.create

    class jax:
        fixed_width: Final = JaxFixedWidthBoundary.create
        piecewise_fixed_width: Final = JaxPiecewiseFixedWidthBoundary.create
