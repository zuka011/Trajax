from typing import cast
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    DataType,
    Distance,
    ControlInputBatch,
    CostFunction,
    JaxCosts,
    JaxObstacleStateProvider,
    JaxObstacleStateSampler,
    JaxDistanceExtractor,
    JaxRiskMetric,
)
from trajax.states import JaxSimpleCosts
from trajax.costs.collision.common import NoMetric

from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims, D

import riskit
import numpy as np
import jax
import jax.numpy as jnp


@jaxtyped
@dataclass(frozen=True)
class JaxDistance[T: int, V: int, M: int, N: int](Distance[T, V, M, N]):
    _array: Float[JaxArray, "T V M N"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, V, M, N]]:
        return np.asarray(self.array)

    @property
    def array(self) -> Float[JaxArray, "T V M N"]:
        return self._array


@dataclass(kw_only=True, frozen=True)
class JaxCollisionCost[
    StateT,
    ObstacleStatesT,
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
    def create[S, OS, SOS, D: JaxDistance, V_: int](
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
            else cast(JaxRiskMetric[S, OS, SOS], NoMetric()),
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> JaxCosts[T, M]:
        def cost(*, states: StateT, samples: SampledObstacleStatesT) -> riskit.JaxCosts:
            return collision_cost(
                distance=self.distance(states=states, obstacle_states=samples).array,
                distance_threshold=self.distance_threshold,
                weight=self.weight,
            )

        return JaxSimpleCosts(
            self.metric.compute(
                cost,
                states=states,
                obstacle_states=self.obstacle_states(),
                sampler=self.sampler,
            )
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
