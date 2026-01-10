from typing import Final

from trajax.predictors.common import (
    StaticPredictor,
    CurvilinearPredictor,
    CovariancePadding,
)
from trajax.predictors.propagators import (
    NumPyLinearCovariancePropagator,
    JaxLinearCovariancePropagator,
)
from trajax.predictors.covariance import (
    NumPyConstantVarianceProvider,
    NumPyConstantCovarianceProvider,
    JaxConstantVarianceProvider,
    JaxConstantCovarianceProvider,
)


class predictor:
    class numpy:
        curvilinear: Final = CurvilinearPredictor.create
        static: Final = StaticPredictor.create

    class jax:
        curvilinear: Final = CurvilinearPredictor.create
        static: Final = StaticPredictor.create


class propagator:
    class numpy:
        linear: Final = NumPyLinearCovariancePropagator.create

        class covariance:
            constant_variance: Final = NumPyConstantVarianceProvider.create
            constant_covariance: Final = NumPyConstantCovarianceProvider.create

        padding: Final = CovariancePadding.create

    class jax:
        linear: Final = JaxLinearCovariancePropagator.create

        class covariance:
            constant_variance: Final = JaxConstantVarianceProvider.create
            constant_covariance: Final = JaxConstantCovarianceProvider.create

        padding: Final = CovariancePadding.create
