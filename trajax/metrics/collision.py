from typing import Any
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    Distance,
    DistanceExtractor,
    StateSequence,
    ObstacleStateSequences,
    SimulationData,
    Metric,
)
from trajax.collectors import access

from numtypes import Array, BoolArray, Dims, shape_of

import numpy as np


@dataclass(kw_only=True, frozen=True)
class CollisionMetricResult[T: int, V: int = int]:
    distances: Array[Dims[T, V]]
    distance_threshold: float

    @cached_property
    def min_distances(self) -> Array[Dims[V]]:
        return self.distances.min(axis=0)

    @cached_property
    def collisions(self) -> BoolArray[Dims[T, V]]:
        return self.distances <= self.distance_threshold

    @cached_property
    def collision_detected(self) -> bool:
        return bool(self.collisions.any())


@dataclass(kw_only=True, frozen=True)
class CollisionMetric[StateBatchT, SampledObstacleStatesT](
    Metric[CollisionMetricResult]
):
    distance: DistanceExtractor[StateBatchT, SampledObstacleStatesT, Distance]
    distance_threshold: float

    @staticmethod
    def create[S, SOS](
        *,
        distance_threshold: float,
        distance: DistanceExtractor[S, SOS, Distance],
    ) -> "CollisionMetric":
        return CollisionMetric(distance=distance, distance_threshold=distance_threshold)

    def compute[T: int = int](self, data: SimulationData) -> CollisionMetricResult[T]:
        states = data(
            access.states.assume(StateSequence[T, Any, StateBatchT]).require()
        )
        obstacle_states = data(
            access.obstacle_states.assume(
                ObstacleStateSequences[T, Any, Any, SampledObstacleStatesT]
            ).require()
        )

        measured_distances = self.distance(
            states=states.batched(), obstacle_states=obstacle_states.single()
        )

        distances = np.asarray(measured_distances).reshape(states.horizon, -1)

        assert shape_of(
            distances, matches=(states.horizon, measured_distances.vehicle_parts)
        )

        return CollisionMetricResult(
            distances=distances, distance_threshold=self.distance_threshold
        )

    @property
    def name(self) -> str:
        return "collision"
