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
from trajax.obstacles.sampler import (
    NumPyGaussianObstacleStateSampler,
    JaxGaussianObstacleStateSampler,
)
from trajax.obstacles.assignment import id_assignment


class obstacles:
    predicting: Final = PredictingObstacleStateProvider.create  # TODO: REmove!
    id_assignment: Final = id_assignment

    class numpy:
        empty: Final = NumPyStaticObstacleStateProvider.empty
        static: Final = NumPyStaticObstacleStateProvider.create
        dynamic: Final = NumPyDynamicObstacleStateProvider.create
        predicting: Final = PredictingObstacleStateProvider.create

        class sampler:
            gaussian: Final = NumPyGaussianObstacleStateSampler.create

    class jax:
        empty: Final = JaxStaticObstacleStateProvider.empty
        static: Final = JaxStaticObstacleStateProvider.create
        dynamic: Final = JaxDynamicObstacleStateProvider.create
        predicting: Final = PredictingObstacleStateProvider.create

        class sampler:
            gaussian: Final = JaxGaussianObstacleStateSampler.create
