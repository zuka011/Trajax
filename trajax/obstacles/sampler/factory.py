from typing import Final

from trajax.obstacles.sampler import (
    NumPyGaussianObstacleStateSampler,
    JaxGaussianObstacleStateSampler,
)


class sampler:
    class numpy:
        gaussian: Final = NumPyGaussianObstacleStateSampler.create

    class jax:
        gaussian: Final = JaxGaussianObstacleStateSampler.create
