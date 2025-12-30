from typing import Final

from trajax.predictors.common import CurvilinearPredictor
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
