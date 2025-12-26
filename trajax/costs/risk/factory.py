from typing import Final

from trajax.costs.risk.common import NoMetric, MeanVarianceMetric


class risk:
    none: Final = NoMetric.create
    mean_variance: Final = MeanVarianceMetric.create
