from typing import Final


from trajax.obstacles.common import PredictingObstacleStateProvider
from trajax.obstacles.static import (
    NumPyStaticObstacleStateProvider,
    JaxStaticObstacleStateProvider,
)
from trajax.obstacles.simulating import (
    NumPyDynamicObstacleStateProvider,
    JaxDynamicObstacleStateProvider,
)
from trajax.obstacles.sampler import sampler
from trajax.obstacles.assignment import id_assignment


class obstacles:
    sampler: Final = sampler
    id_assignment: Final = id_assignment
    predicting: Final = PredictingObstacleStateProvider.create

    class numpy:
        empty: Final = NumPyStaticObstacleStateProvider.empty
        static: Final = NumPyStaticObstacleStateProvider.create
        dynamic: Final = NumPyDynamicObstacleStateProvider.create

    class jax:
        empty: Final = JaxStaticObstacleStateProvider.empty
        static: Final = JaxStaticObstacleStateProvider.create
        dynamic: Final = JaxDynamicObstacleStateProvider.create
