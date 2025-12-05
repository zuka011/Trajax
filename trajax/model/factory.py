from typing import Final

from trajax.bicycle import NumPyBicycleModel, JaxBicycleModel


class model:
    class kinematic_bicycle:
        numpy: Final = NumPyBicycleModel.create
        jax: Final = JaxBicycleModel.create
