from typing import Protocol

from trajax.types.costs.collision.common import (
    ObstacleStateProvider,
    ObstacleStateSampler,
    DistanceExtractor,
    SampleCostFunction,
)

from numtypes import Array, Dims


class NumPyObstacleStateProvider[ObstacleStatesT](
    ObstacleStateProvider[ObstacleStatesT], Protocol
): ...


class NumPyObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT](
    ObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT], Protocol
): ...


class NumPyDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT](
    DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT], Protocol
): ...


class NumPyRiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT](Protocol):
    def compute[T: int, M: int](
        self,
        cost_function: SampleCostFunction[
            StateBatchT, SampledObstacleStatesT, Array[Dims[T, M, int]]
        ],
        *,
        states: StateBatchT,
        obstacle_states: ObstacleStatesT,
        sampler: NumPyObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT],
    ) -> Array[Dims[T, M]]:
        """Computes the risk metric based on the provided cost function and returns it as a NumPy array."""
        ...
