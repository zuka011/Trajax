from typing import Final

from trajax.obstacles.sampler import (
    NumPyObstaclePositionAndHeadingSampler,
    JaxObstaclePositionAndHeadingSampler,
)


class sampler:
    class numpy:
        position_and_heading: Final = NumPyObstaclePositionAndHeadingSampler

    class jax:
        position_and_heading: Final = JaxObstaclePositionAndHeadingSampler
