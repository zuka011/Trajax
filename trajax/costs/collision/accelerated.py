from typing import cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    jaxtyped,
    DataType,
    Distance,
    ControlInputBatch,
    CostFunction,
    ObstacleStates,
    ObstacleStateSampler,
    SampleCostFunction,
    JaxCosts,
    JaxObstacleStateProvider,
    JaxObstacleStateSampler,
    JaxDistanceExtractor,
    JaxRisk,
    JaxRiskMetric,
)
from trajax.states import JaxSimpleCosts

from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims, D

import numpy as np
import jax
import jax.numpy as jnp


@jaxtyped
@dataclass(frozen=True)
class JaxDistance[T: int, V: int, M: int, N: int](Distance[T, V, M, N]):
    _array: Float[JaxArray, "T V M N"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, V, M, N]]:
        return self._numpy_array

    @property
    def horizon(self) -> T:
        return cast(T, self._array.shape[0])

    @property
    def vehicle_parts(self) -> V:
        return cast(V, self._array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self._array.shape[2])

    @property
    def sample_count(self) -> N:
        return cast(N, self._array.shape[3])

    @property
    def array(self) -> Float[JaxArray, "T V M N"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, V, M, N]]:
        return np.asarray(self._array)


class JaxNoMetric:
    @staticmethod
    def create() -> "JaxNoMetric":
        return JaxNoMetric()

    def compute[StateT, ObstacleStateT, SampledObstacleStateT](
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Float[JaxArray, "T M N"]
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: ObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> JaxRisk:
        samples = sampler(obstacle_states, count=1)
        return JaxRisk(cost_function(states=states, samples=samples).squeeze(axis=-1))

    @property
    def name(self) -> str:
        return "No Metric"


@dataclass(kw_only=True, frozen=True)
class JaxCollisionCost[
    StateT,
    ObstacleStatesT: ObstacleStates,
    SampledObstacleStatesT,
    DistanceT: JaxDistance,
    V: int,
](CostFunction[ControlInputBatch, StateT, JaxCosts]):
    obstacle_states: JaxObstacleStateProvider[ObstacleStatesT]
    sampler: JaxObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT]
    distance: JaxDistanceExtractor[StateT, SampledObstacleStatesT, DistanceT]
    distance_threshold: Float[JaxArray, "V"]
    weight: float
    metric: JaxRiskMetric[StateT, ObstacleStatesT, SampledObstacleStatesT]

    @staticmethod
    def create[S, OS: ObstacleStates, SOS, D: JaxDistance, V_: int](
        *,
        obstacle_states: JaxObstacleStateProvider[OS],
        sampler: JaxObstacleStateSampler[OS, SOS],
        distance: JaxDistanceExtractor[S, SOS, D],
        distance_threshold: Array[Dims[V_]],
        weight: float,
        metric: JaxRiskMetric[S, OS, SOS] | None = None,
    ) -> "JaxCollisionCost[S, OS, SOS, D, V_]":
        return JaxCollisionCost(
            obstacle_states=obstacle_states,
            sampler=sampler,
            distance=distance,
            distance_threshold=jnp.asarray(distance_threshold),
            weight=weight,
            metric=metric
            if metric is not None
            else cast(JaxRiskMetric[S, OS, SOS], JaxNoMetric()),
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> JaxCosts[T, M]:
        def cost(
            *, states: StateT, samples: SampledObstacleStatesT
        ) -> Float[JaxArray, "T M N"]:
            return collision_cost(
                distance=self.distance(states=states, obstacle_states=samples).array,
                distance_threshold=self.distance_threshold,
                weight=self.weight,
            )

        return JaxSimpleCosts(
            jnp.zeros((inputs.horizon, inputs.rollout_count))
            if (obstacle_states := self.obstacle_states()).count == 0
            else self.metric.compute(
                cost,
                states=states,
                obstacle_states=obstacle_states,
                sampler=self.sampler,
            ).array
        )


@jax.jit
@jaxtyped
def collision_cost(
    *,
    distance: Float[JaxArray, "T V M N"],
    distance_threshold: Float[JaxArray, "V"],
    weight: Scalar,
) -> Float[JaxArray, "T M N"]:
    cost = distance_threshold[jnp.newaxis, :, jnp.newaxis, jnp.newaxis] - distance
    return weight * jnp.clip(cost, 0, None).sum(axis=1)
