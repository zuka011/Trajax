from trajax.costs import (
    NumPyObstacleStates,
    NumPySampledObstacleStates,
    NumPyObstacleStateSampler,
)

import numpy as np


class NumPyObstaclePositionAndHeadingSampler[StateT: NumPyObstacleStates](
    NumPyObstacleStateSampler[StateT, NumPySampledObstacleStates]
):
    def __call__(self, states: StateT, *, count: int) -> NumPySampledObstacleStates:
        return states.sampled(
            x=states.x()[..., np.newaxis],
            y=states.y()[..., np.newaxis],
            heading=states.heading()[..., np.newaxis],
        )

        # mean = np.stack([states.x(), states.y(), states.heading()], axis=1)
        # flat_mean = mean.transpose(0, 2, 1).reshape(-1, 3)
        # return states.sampled(
        #     x=distribution.numpy.gaussian(
        #         mean=states.x(),
        #     )
        #     y=np.repeat(
        #         states.y()[:, :, np.newaxis],
        #         repeats=count,
        #         axis=2,
        #     ),
        #     heading=np.repeat(
        #         states.heading()[:, :, np.newaxis],
        #         repeats=count,
        #         axis=2,
        #     ),
        # )
