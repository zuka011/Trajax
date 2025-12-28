from typing import overload

from trajax.types import ObstacleStateSampler, SampleCostFunction

from numtypes import Array, Dims
from jaxtyping import Array as JaxArray


class NoMetric:
    @staticmethod
    def create() -> "NoMetric":
        return NoMetric()

    @overload
    def compute[StateT, ObstacleStateT, SampledObstacleStateT, T: int, M: int](
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Array[Dims[T, M, int]]
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: ObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> Array[Dims[T, M]]: ...

    @overload
    def compute[StateT, ObstacleStateT, SampledObstacleStateT](
        self,
        cost_function: SampleCostFunction[StateT, SampledObstacleStateT, JaxArray],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: ObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> JaxArray: ...

    def compute[StateT, ObstacleStateT, SampledObstacleStateT, T: int, M: int](
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Array[Dims[T, M, int]] | JaxArray
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: ObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> Array[Dims[T, M]] | JaxArray:
        samples = sampler(obstacle_states, count=1)
        return cost_function(states=states, samples=samples).squeeze(axis=-1)
