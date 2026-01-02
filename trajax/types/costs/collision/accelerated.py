from typing import Protocol, cast
from dataclasses import dataclass

from trajax.types.array import jaxtyped, DataType
from trajax.types.costs.collision.common import (
    ObstacleStateProvider,
    ObstacleStateSampler,
    DistanceExtractor,
    SampleCostFunction,
)

from numtypes import Array, Dims
from jaxtyping import Float, Array as JaxArray

import numpy as np


class JaxObstacleStateProvider[ObstacleStatesT](
    ObstacleStateProvider[ObstacleStatesT], Protocol
): ...


class JaxObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT](
    ObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT], Protocol
): ...


class JaxDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT](
    DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT], Protocol
): ...


@jaxtyped
@dataclass(frozen=True)
class JaxRisk[T: int, M: int]:
    _array: Float[JaxArray, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return np.asarray(self.array)

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[1])

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        return self._array


class JaxRiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT](Protocol):
    def compute(
        self,
        cost_function: SampleCostFunction[
            StateBatchT, SampledObstacleStatesT, Float[JaxArray, "T M N"]
        ],
        *,
        states: StateBatchT,
        obstacle_states: ObstacleStatesT,
        sampler: JaxObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT],
    ) -> JaxRisk:
        """Computes the risk metric based on the provided cost function and returns it as a JAX array."""
        ...
