from typing import Protocol

from trajax.types.costs.collision.common import (
    ObstacleStateProvider,
    ObstacleStateSampler,
    DistanceExtractor,
    SampleCostFunction,
)

from jaxtyping import Float, Array as JaxArray


class JaxObstacleStateProvider[ObstacleStatesT](
    ObstacleStateProvider[ObstacleStatesT], Protocol
): ...


class JaxObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT](
    ObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT], Protocol
): ...


class JaxDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT](
    DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT], Protocol
): ...


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
    ) -> Float[JaxArray, "T M"]:
        """Computes the risk metric based on the provided cost function and returns it as a JAX array."""
        ...
