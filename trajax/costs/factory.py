from typing import Final

from trajax.costs.basic import (
    ContouringCost as NumPyContouringCost,
    LagCost as NumPyLagCost,
)
from trajax.costs.accelerated import (
    ContouringCost as JaxContouringCost,
    LagCost as JaxLagCost,
)


class costs:
    class numpy:
        class tracking:
            contouring: Final = NumPyContouringCost.create
            lag: Final = NumPyLagCost.create

    class jax:
        class tracking:
            contouring: Final = JaxContouringCost.create
            lag: Final = JaxLagCost.create
