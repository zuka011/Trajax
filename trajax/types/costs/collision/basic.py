from typing import Protocol
from dataclasses import dataclass

from trajax.types.array import DataType
from trajax.types.costs.collision.common import (
    ObstacleStateProvider,
    ObstacleStateSampler,
    DistanceExtractor,
    SampleCostFunction,
    Risk,
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


@dataclass(frozen=True)
class NumPyRisk[T: int, M: int](Risk[T, M]):
    _array: Array[Dims[T, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return self._array

    @property
    def horizon(self) -> T:
        return self._array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self._array.shape[1]

    @property
    def array(self) -> Array[Dims[T, M]]:
        return self._array


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
    ) -> NumPyRisk[T, M]:
        """Computes the risk metric based on the provided cost function and returns it as a NumPy array."""
        ...
