from typing import Final

from trajax.predictors.common import CurvilinearPredictor
from trajax.predictors.propagators import NumPyLinearCovariancePropagator


class predictor:
    curvilinear: Final = CurvilinearPredictor.create


class propagator:
    class numpy:
        linear: Final = NumPyLinearCovariancePropagator.create
