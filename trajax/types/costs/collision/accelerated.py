from typing import Protocol

from trajax.types.costs.collision.common import (
    SampledObstacleStates,
    ObstacleStates,
    ObstacleStateProvider,
    ObstacleStateSampler,
    DistanceExtractor,
    SampleCostFunction,
)

from numtypes import D
from jaxtyping import Float, Array as JaxArray


class JaxSampledObstacleStates[T: int, K: int, N: int](
    SampledObstacleStates[T, K, N], Protocol
):
    @property
    def x_array(self) -> Float[JaxArray, "T K N"]:
        """Returns the x positions of obstacles over time and samples."""
        ...

    @property
    def y_array(self) -> Float[JaxArray, "T K N"]:
        """Returns the y positions of obstacles over time and samples."""
        ...

    @property
    def heading_array(self) -> Float[JaxArray, "T K N"]:
        """Returns the headings of obstacles over time and samples."""
        ...

    @property
    def sample_count(self) -> N:
        """Returns the number of samples per obstacle."""
        ...


class JaxObstacleStates[T: int, K: int](
    ObstacleStates[T, K, JaxSampledObstacleStates[T, K, D[1]]], Protocol
):
    def sampled[N: int = int](
        *,
        x: Float[JaxArray, "T K N"],
        y: Float[JaxArray, "T K N"],
        heading: Float[JaxArray, "T K N"],
        sample_count: N | None = None,
    ) -> JaxSampledObstacleStates[T, K, N]:
        """Returns sampled states of obstacles over time."""
        ...

    @property
    def x_array(self) -> Float[JaxArray, "T K"]:
        """Returns the x positions of obstacles over time."""
        ...

    @property
    def y_array(self) -> Float[JaxArray, "T K"]:
        """Returns the y positions of obstacles over time."""
        ...

    @property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        """Returns the headings of obstacles over time."""
        ...

    @property
    def covariance_array(self) -> Float[JaxArray, "T D_o D_o K"] | None:
        """Returns the covariance matrices of obstacles over time."""
        ...


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
