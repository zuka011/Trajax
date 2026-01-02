from typing import cast
from dataclasses import dataclass

from trajax.types import (
    DataType,
    Distance,
    ControlInputBatch,
    CostFunction,
    ObstacleStateSampler,
    SampleCostFunction,
    NumPyCosts,
    NumPyObstacleStateProvider,
    NumPyObstacleStateSampler,
    NumPyDistanceExtractor,
    NumPyRisk,
    NumPyRiskMetric,
)
from trajax.states import NumPySimpleCosts

from numtypes import Array, Dims, D

import numpy as np


@dataclass(frozen=True)
class NumPyDistance[T: int, V: int, M: int, N: int](Distance[T, V, M, N]):
    array: Array[Dims[T, V, M, N]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, V, M, N]]:
        return self.array


class NumPyNoMetric:
    @staticmethod
    def create() -> "NumPyNoMetric":
        return NumPyNoMetric()

    def compute[StateT, ObstacleStateT, SampledObstacleStateT, T: int, M: int](
        self,
        cost_function: SampleCostFunction[
            StateT, SampledObstacleStateT, Array[Dims[T, M, int]]
        ],
        *,
        states: StateT,
        obstacle_states: ObstacleStateT,
        sampler: ObstacleStateSampler[ObstacleStateT, SampledObstacleStateT],
    ) -> NumPyRisk[T, M]:
        samples = sampler(obstacle_states, count=1)
        return NumPyRisk(cost_function(states=states, samples=samples).squeeze(axis=-1))


@dataclass(kw_only=True, frozen=True)
class NumPyCollisionCost[
    StateT,
    ObstacleStatesT,
    SampledObstacleStatesT,
    DistanceT: NumPyDistance,
    V: int,
](CostFunction[ControlInputBatch, StateT, NumPyCosts]):
    obstacle_states: NumPyObstacleStateProvider[ObstacleStatesT]
    sampler: NumPyObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT]
    distance: NumPyDistanceExtractor[StateT, SampledObstacleStatesT, DistanceT]
    distance_threshold: Array[Dims[V]]
    weight: float
    metric: NumPyRiskMetric[StateT, ObstacleStatesT, SampledObstacleStatesT]

    @staticmethod
    def create[S, OS, SOS, D: NumPyDistance, V_: int](
        *,
        obstacle_states: NumPyObstacleStateProvider[OS],
        sampler: NumPyObstacleStateSampler[OS, SOS],
        distance: NumPyDistanceExtractor[S, SOS, D],
        distance_threshold: Array[Dims[V_]],
        weight: float,
        metric: NumPyRiskMetric[S, OS, SOS] | None = None,
    ) -> "NumPyCollisionCost[S, OS, SOS, D, V_]":
        return NumPyCollisionCost(
            obstacle_states=obstacle_states,
            sampler=sampler,
            distance=distance,
            distance_threshold=distance_threshold,
            weight=weight,
            metric=metric
            if metric is not None
            else cast(NumPyRiskMetric[S, OS, SOS], NumPyNoMetric()),
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> NumPyCosts[T, M]:
        def cost(
            *, states: StateT, samples: SampledObstacleStatesT
        ) -> Array[Dims[T, M, int]]:
            cost = (
                self.distance_threshold[np.newaxis, :, np.newaxis, np.newaxis]
                - self.distance(states=states, obstacle_states=samples).array
            )

            return self.weight * np.clip(cost, 0, None).sum(axis=1)

        return NumPySimpleCosts(
            self.metric.compute(
                cost,
                states=states,
                obstacle_states=self.obstacle_states(),
                sampler=self.sampler,
            ).array
        )
