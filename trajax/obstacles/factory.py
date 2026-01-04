from typing import Final

from trajax.types import (
    ObstacleMotionPredictor,
    NumPyObstacleStatesHistory,
    JaxObstacleStatesHistory,
    D_o,
)

from trajax.obstacles.common import PredictingObstacleStateProvider
from trajax.obstacles.basic import (
    NumPyObstacleStatesForTimeStep,
    NumPyObstacleStatesRunningHistory,
)
from trajax.obstacles.accelerated import (
    JaxObstacleStatesForTimeStep,
    JaxObstacleStatesRunningHistory,
)
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

    class numpy:
        empty: Final = NumPyStaticObstacleStateProvider.empty
        static: Final = NumPyStaticObstacleStateProvider.create
        dynamic: Final = NumPyDynamicObstacleStateProvider.create

        @staticmethod
        def predicting[PredictionT](
            *,
            predictor: ObstacleMotionPredictor[
                NumPyObstacleStatesHistory[int, D_o, int], PredictionT
            ],
        ) -> PredictingObstacleStateProvider[
            NumPyObstacleStatesForTimeStep[int],
            NumPyObstacleStatesRunningHistory[int],
            PredictionT,
        ]:
            return PredictingObstacleStateProvider.create(
                predictor=predictor,
                history=NumPyObstacleStatesRunningHistory.empty(),
            )

    class jax:
        empty: Final = JaxStaticObstacleStateProvider.empty
        static: Final = JaxStaticObstacleStateProvider.create
        dynamic: Final = JaxDynamicObstacleStateProvider.create

        @staticmethod
        def predicting[PredictionT](
            *,
            predictor: ObstacleMotionPredictor[
                JaxObstacleStatesHistory[int, D_o, int], PredictionT
            ],
        ) -> PredictingObstacleStateProvider[
            JaxObstacleStatesForTimeStep[int],
            JaxObstacleStatesRunningHistory[int],
            PredictionT,
        ]:
            return PredictingObstacleStateProvider.create(
                predictor=predictor,
                history=JaxObstacleStatesRunningHistory.empty(),
            )
