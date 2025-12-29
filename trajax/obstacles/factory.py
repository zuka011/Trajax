from typing import Final

from trajax.obstacles.basic import (
    NumPyStaticObstacleStateProvider,
    NumPyDynamicObstacleStateProvider,
)
from trajax.obstacles.accelerated import JaxStaticObstacleStateProvider
from trajax.obstacles.sampler import sampler


class obstacles:
    sampler: Final = sampler

    class numpy:
        empty: Final = NumPyStaticObstacleStateProvider.empty
        static: Final = NumPyStaticObstacleStateProvider.create
        dynamic: Final = NumPyDynamicObstacleStateProvider.create

    class jax:
        empty: Final = JaxStaticObstacleStateProvider.empty
        static: Final = JaxStaticObstacleStateProvider.create
