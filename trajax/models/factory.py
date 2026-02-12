from typing import Final

from trajax.models.integrator import (
    NumPyIntegratorModel,
    NumPyIntegratorObstacleModel,
    JaxIntegratorModel,
    JaxIntegratorObstacleModel,
)
from trajax.models.bicycle import (
    NumPyBicycleModel,
    NumPyBicycleObstacleModel,
    NumPyBicyclePoseCovarianceExtractor,
    NumPyBicyclePositionCovarianceExtractor,
    JaxBicycleModel,
    JaxBicycleObstacleModel,
    JaxBicyclePoseCovarianceExtractor,
    JaxBicyclePositionCovarianceExtractor,
)
from trajax.models.unicycle import (
    NumPyUnicycleModel,
    NumPyUnicycleObstacleModel,
    JaxUnicycleModel,
    JaxUnicycleObstacleModel,
)


class model:
    class numpy:
        class integrator:
            dynamical: Final = NumPyIntegratorModel.create
            obstacle: Final = NumPyIntegratorObstacleModel.create

        class bicycle:
            dynamical: Final = NumPyBicycleModel.create
            obstacle: Final = NumPyBicycleObstacleModel.create

            class covariance_of:
                pose: Final = NumPyBicyclePoseCovarianceExtractor
                position: Final = NumPyBicyclePositionCovarianceExtractor

        class unicycle:
            dynamical: Final = NumPyUnicycleModel.create
            obstacle: Final = NumPyUnicycleObstacleModel.create

    class jax:
        class integrator:
            dynamical: Final = JaxIntegratorModel.create
            obstacle: Final = JaxIntegratorObstacleModel.create

        class bicycle:
            dynamical: Final = JaxBicycleModel.create
            obstacle: Final = JaxBicycleObstacleModel.create

            class covariance_of:
                pose: Final = JaxBicyclePoseCovarianceExtractor
                position: Final = JaxBicyclePositionCovarianceExtractor

        class unicycle:
            dynamical: Final = JaxUnicycleModel.create
            obstacle: Final = JaxUnicycleObstacleModel.create
