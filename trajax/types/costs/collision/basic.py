from typing import Protocol

from trajax.types.costs.collision.common import (
    ObstacleStateProvider,
    ObstacleStateSampler,
    DistanceExtractor,
    SampleCostFunction,
)

from numtypes import Array, Dims


class NumPyObstacleStateProvider[StateT](ObstacleStateProvider[StateT], Protocol): ...


class NumPyObstacleStateSampler[StateT, SampleT](
    ObstacleStateSampler[StateT, SampleT], Protocol
): ...


class NumPyDistanceExtractor[StateT, SampleT, DistanceT](
    DistanceExtractor[StateT, SampleT, DistanceT], Protocol
): ...


class NumPyRiskMetric[StateT, ObstacleStateT, SampledObstacleStateT](Protocol):
    def compute[T: int, M: int](
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Array[Dims[T, M, int]]
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: NumPyObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> Array[Dims[T, M]]:
        """Computes the risk metric based on the provided cost function and returns it as a NumPy array."""
        ...
