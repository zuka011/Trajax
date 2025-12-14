from typing import Final

from trajax.costs.basic import (
    ContouringCost as NumPyContouringCost,
    LagCost as NumPyLagCost,
    ProgressCost as NumPyProgressCost,
)
from trajax.costs.accelerated import (
    ContouringCost as JaxContouringCost,
    LagCost as JaxLagCost,
    ProgressCost as JaxProgressCost,
)


class costs:
    class numpy:
        class tracking:
            contouring: Final = NumPyContouringCost.create
            lag: Final = NumPyLagCost.create
            progress: Final = NumPyProgressCost.create

    class jax:
        class tracking:
            contouring: Final = JaxContouringCost.create
            lag: Final = JaxLagCost.create
            progress: Final = JaxProgressCost.create
