from typing import Final

from trajax.integrator import NumPyIntegratorModel, JaxIntegratorModel
from trajax.bicycle import NumPyBicycleModel, JaxBicycleModel


class model:
    class numpy:
        integrator: Final = NumPyIntegratorModel.create
        kinematic_bicycle: Final = NumPyBicycleModel.create

    class jax:
        integrator: Final = JaxIntegratorModel.create
        kinematic_bicycle: Final = JaxBicycleModel.create
