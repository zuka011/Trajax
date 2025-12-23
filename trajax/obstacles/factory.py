from typing import Final

from trajax.obstacles.basic import NumPyStaticObstacleStateProvider
from trajax.obstacles.accelerated import JaxStaticObstacleStateProvider


class obstacles:
    class numpy:
        empty: Final = NumPyStaticObstacleStateProvider.empty
        static: Final = NumPyStaticObstacleStateProvider.create

    class jax:
        empty: Final = JaxStaticObstacleStateProvider.empty
        static: Final = JaxStaticObstacleStateProvider.create
