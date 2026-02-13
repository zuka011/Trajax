from typing import Final

from trajax.models.integrator import (
    NumPyIntegratorModel,
    NumPyIntegratorObstacleModel,
    NumPyFiniteDifferenceIntegratorStateEstimator,
    JaxIntegratorModel,
    JaxIntegratorObstacleModel,
    JaxFiniteDifferenceIntegratorStateEstimator,
)
from trajax.models.bicycle import (
    NumPyBicycleModel,
    NumPyBicycleObstacleModel,
    NumPyBicyclePoseCovarianceExtractor,
    NumPyBicyclePositionCovarianceExtractor,
    NumPyFiniteDifferenceBicycleStateEstimator,
    JaxBicycleModel,
    JaxBicycleObstacleModel,
    JaxBicyclePoseCovarianceExtractor,
    JaxBicyclePositionCovarianceExtractor,
    JaxFiniteDifferenceBicycleStateEstimator,
)
from trajax.models.unicycle import (
    NumPyUnicycleModel,
    NumPyUnicycleObstacleModel,
    NumPyFiniteDifferenceUnicycleStateEstimator,
    JaxUnicycleModel,
    JaxUnicycleObstacleModel,
    JaxFiniteDifferenceUnicycleStateEstimator,
)


class model:
    class numpy:
        class integrator:
            dynamical: Final = NumPyIntegratorModel.create
            obstacle: Final = NumPyIntegratorObstacleModel.create

            class estimator:
                finite_difference: Final = (
                    NumPyFiniteDifferenceIntegratorStateEstimator.create
                )

        class bicycle:
            dynamical: Final = NumPyBicycleModel.create
            obstacle: Final = NumPyBicycleObstacleModel.create

            class estimator:
                finite_difference: Final = (
                    NumPyFiniteDifferenceBicycleStateEstimator.create
                )

            class covariance_of:
                pose: Final = NumPyBicyclePoseCovarianceExtractor
                position: Final = NumPyBicyclePositionCovarianceExtractor

        class unicycle:
            dynamical: Final = NumPyUnicycleModel.create
            obstacle: Final = NumPyUnicycleObstacleModel.create

            class estimator:
                finite_difference: Final = (
                    NumPyFiniteDifferenceUnicycleStateEstimator.create
                )

    class jax:
        class integrator:
            dynamical: Final = JaxIntegratorModel.create
            obstacle: Final = JaxIntegratorObstacleModel.create

            class estimator:
                finite_difference: Final = (
                    JaxFiniteDifferenceIntegratorStateEstimator.create
                )

        class bicycle:
            dynamical: Final = JaxBicycleModel.create
            obstacle: Final = JaxBicycleObstacleModel.create

            class estimator:
                finite_difference: Final = (
                    JaxFiniteDifferenceBicycleStateEstimator.create
                )

            class covariance_of:
                pose: Final = JaxBicyclePoseCovarianceExtractor
                position: Final = JaxBicyclePositionCovarianceExtractor

        class unicycle:
            dynamical: Final = JaxUnicycleModel.create
            obstacle: Final = JaxUnicycleObstacleModel.create

            class estimator:
                finite_difference: Final = (
                    JaxFiniteDifferenceUnicycleStateEstimator.create
                )
