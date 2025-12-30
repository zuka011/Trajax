from typing import Final

from trajax.predictors.common import CurvilinearPredictor, CovariancePadding
from trajax.predictors.propagators import (
    NumPyLinearCovariancePropagator,
    JaxLinearCovariancePropagator,
)
from trajax.predictors.covariance import (
    NumPyConstantVarianceProvider,
    JaxConstantVarianceProvider,
)


class predictor:
    curvilinear: Final = CurvilinearPredictor.create


class propagator:
    padding: Final = CovariancePadding.create

    class numpy:
        linear: Final = NumPyLinearCovariancePropagator.create

        class covariance:
            constant_variance: Final = NumPyConstantVarianceProvider

    class jax:
        linear: Final = JaxLinearCovariancePropagator.create

        class covariance:
            constant_variance: Final = JaxConstantVarianceProvider
