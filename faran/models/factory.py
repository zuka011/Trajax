from typing import Final

from faran.models.integrator import (
    NumPyIntegratorModel,
    NumPyIntegratorObstacleModel,
    NumPyFiniteDifferenceIntegratorStateEstimator,
    NumPyKfIntegratorStateEstimator,
    JaxIntegratorModel,
    JaxIntegratorObstacleModel,
    JaxFiniteDifferenceIntegratorStateEstimator,
    JaxKfIntegratorStateEstimator,
)
from faran.models.bicycle import (
    NumPyBicycleModel,
    NumPyBicycleObstacleModel,
    NumPyFiniteDifferenceBicycleStateEstimator,
    NumPyKfBicycleStateEstimator,
    JaxBicycleModel,
    JaxBicycleObstacleModel,
    JaxFiniteDifferenceBicycleStateEstimator,
    JaxKfBicycleStateEstimator,
)
from faran.models.unicycle import (
    NumPyUnicycleModel,
    NumPyUnicycleObstacleModel,
    NumPyFiniteDifferenceUnicycleStateEstimator,
    NumPyKfUnicycleStateEstimator,
    JaxUnicycleModel,
    JaxUnicycleObstacleModel,
    JaxFiniteDifferenceUnicycleStateEstimator,
    JaxKfUnicycleStateEstimator,
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
                kf: Final = NumPyKfIntegratorStateEstimator.create

        class bicycle:
            dynamical: Final = NumPyBicycleModel.create
            obstacle: Final = NumPyBicycleObstacleModel.unscented

            class estimator:
                finite_difference: Final = (
                    NumPyFiniteDifferenceBicycleStateEstimator.create
                )
                ekf: Final = NumPyKfBicycleStateEstimator.ekf
                ukf: Final = NumPyKfBicycleStateEstimator.ukf

        class unicycle:
            dynamical: Final = NumPyUnicycleModel.create
            obstacle: Final = NumPyUnicycleObstacleModel.unscented

            class estimator:
                finite_difference: Final = (
                    NumPyFiniteDifferenceUnicycleStateEstimator.create
                )
                ekf: Final = NumPyKfUnicycleStateEstimator.ekf
                ukf: Final = NumPyKfUnicycleStateEstimator.ukf

    class jax:
        class integrator:
            dynamical: Final = JaxIntegratorModel.create
            obstacle: Final = JaxIntegratorObstacleModel.create

            class estimator:
                finite_difference: Final = (
                    JaxFiniteDifferenceIntegratorStateEstimator.create
                )
                kf: Final = JaxKfIntegratorStateEstimator.create

        class bicycle:
            dynamical: Final = JaxBicycleModel.create
            obstacle: Final = JaxBicycleObstacleModel.unscented

            class estimator:
                finite_difference: Final = (
                    JaxFiniteDifferenceBicycleStateEstimator.create
                )
                ekf: Final = JaxKfBicycleStateEstimator.ekf
                ukf: Final = JaxKfBicycleStateEstimator.ukf

        class unicycle:
            dynamical: Final = JaxUnicycleModel.create
            obstacle: Final = JaxUnicycleObstacleModel.unscented

            class estimator:
                finite_difference: Final = (
                    JaxFiniteDifferenceUnicycleStateEstimator.create
                )
                ekf: Final = JaxKfUnicycleStateEstimator.ekf
                ukf: Final = JaxKfUnicycleStateEstimator.ukf
