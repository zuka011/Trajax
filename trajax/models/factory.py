from typing import Final

from trajax.models.integrator import (
    NumPyIntegratorModel,
    NumPyIntegratorObstacleModel,
    JaxIntegratorModel,
)
from trajax.models.bicycle import (
    NumPyBicycleModel,
    NumPyBicycleObstacleModel,
    JaxBicycleModel,
)


class model:
    class numpy:
        class integrator:
            dynamical: Final = NumPyIntegratorModel.create
            obstacle: Final = NumPyIntegratorObstacleModel.create

        class kinematic_bicycle:
            dynamical: Final = NumPyBicycleModel.create
            obstacle: Final = NumPyBicycleObstacleModel.create

    class jax:
        integrator: Final = JaxIntegratorModel.create
        kinematic_bicycle: Final = JaxBicycleModel.create
