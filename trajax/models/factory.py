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
            estimator: Final = NumPyFiniteDifferenceIntegratorStateEstimator.create

        class bicycle:
            dynamical: Final = NumPyBicycleModel.create
            obstacle: Final = NumPyBicycleObstacleModel.create
            estimator: Final = NumPyFiniteDifferenceBicycleStateEstimator.create

            class covariance_of:
                pose: Final = NumPyBicyclePoseCovarianceExtractor
                position: Final = NumPyBicyclePositionCovarianceExtractor

        class unicycle:
            dynamical: Final = NumPyUnicycleModel.create
            obstacle: Final = NumPyUnicycleObstacleModel.create
            estimator: Final = NumPyFiniteDifferenceUnicycleStateEstimator.create

    class jax:
        class integrator:
            dynamical: Final = JaxIntegratorModel.create
            obstacle: Final = JaxIntegratorObstacleModel.create
            estimator: Final = JaxFiniteDifferenceIntegratorStateEstimator.create

        class bicycle:
            dynamical: Final = JaxBicycleModel.create
            obstacle: Final = JaxBicycleObstacleModel.create
            estimator: Final = JaxFiniteDifferenceBicycleStateEstimator.create

            class covariance_of:
                pose: Final = JaxBicyclePoseCovarianceExtractor
                position: Final = JaxBicyclePositionCovarianceExtractor

        class unicycle:
            dynamical: Final = JaxUnicycleModel.create
            obstacle: Final = JaxUnicycleObstacleModel.create
            estimator: Final = JaxFiniteDifferenceUnicycleStateEstimator.create
