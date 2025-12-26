from typing import Final

from trajax.costs.collision import NoMetric
from trajax.costs.risk.basic import NumPyMeanVarianceMetric
from trajax.costs.risk.accelerated import JaxMeanVarianceMetric


class risk:
    none: Final = NoMetric.create

    class numpy:
        mean_variance: Final = NumPyMeanVarianceMetric.create

    class jax:
        mean_variance: Final = JaxMeanVarianceMetric.create
