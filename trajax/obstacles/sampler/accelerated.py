from trajax.costs import (
    JaxObstacleStates,
    JaxSampledObstacleStates,
    JaxObstacleStateSampler,
)


class JaxObstaclePositionAndHeadingSampler[StateT: JaxObstacleStates](
    JaxObstacleStateSampler[StateT, JaxSampledObstacleStates]
):
    def __call__(self, states: StateT, *, count: int) -> JaxSampledObstacleStates:
        return states.single()
