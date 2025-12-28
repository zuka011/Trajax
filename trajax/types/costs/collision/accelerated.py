from typing import Protocol

from trajax.types.costs.collision.common import (
    ObstacleStateProvider,
    ObstacleStateSampler,
    DistanceExtractor,
    SampleCostFunction,
)

from jaxtyping import Float, Array as JaxArray


class JaxObstacleStateProvider[StateT](ObstacleStateProvider[StateT], Protocol): ...


class JaxObstacleStateSampler[StateT, SampleT](
    ObstacleStateSampler[StateT, SampleT], Protocol
): ...


class JaxDistanceExtractor[StateT, SampleT, DistanceT](
    DistanceExtractor[StateT, SampleT, DistanceT], Protocol
): ...


class JaxRiskMetric[StateT, ObstacleStateT, SampledObstacleStateT](Protocol):
    def compute(
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Float[JaxArray, "T M N"]
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: JaxObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> Float[JaxArray, "T M"]:
        """Computes the risk metric based on the provided cost function and returns it as a JAX array."""
        ...
