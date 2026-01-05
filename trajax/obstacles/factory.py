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


class obstacles:
    sampler: Final = sampler
    predicting: Final = PredictingObstacleStateProvider.create

    class numpy:
        empty: Final = NumPyStaticObstacleStateProvider.empty
        static: Final = NumPyStaticObstacleStateProvider.create
        dynamic: Final = NumPyDynamicObstacleStateProvider.create

    class jax:
        empty: Final = JaxStaticObstacleStateProvider.empty
        static: Final = JaxStaticObstacleStateProvider.create
        dynamic: Final = JaxDynamicObstacleStateProvider.create
