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


class costs:
    class numpy:
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
