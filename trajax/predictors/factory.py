from typing import Final

from trajax.predictors.common import CurvilinearPredictor, CovariancePadding
from trajax.predictors.propagators import (
    NumPyLinearCovariancePropagator,
    JaxLinearCovariancePropagator,
)


class predictor:
    curvilinear: Final = CurvilinearPredictor.create


class propagator:
    class numpy:
        linear: Final = NumPyLinearCovariancePropagator.create

    class jax:
        linear: Final = JaxLinearCovariancePropagator.create

    padding: Final = CovariancePadding.create
