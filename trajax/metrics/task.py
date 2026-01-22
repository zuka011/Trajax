from typing import Any
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    StateSequence,
    SimulationData,
    Metric,
    Positions,
    PositionExtractor,
    Trajectory,
)
from trajax.collectors import access

from numtypes import BoolArray, Array, Dims, D, array

import numpy as np


@dataclass(kw_only=True, frozen=True)
class TaskCompletionMetricResult[T: int = int]:
    completion: BoolArray[Dims[T]]
    completion_time: float
    stretch: float

    @cached_property
    def completed(self) -> bool:
        return bool(self.completion.any())


@dataclass(kw_only=True, frozen=True)
class TaskCompletionMetric[StateBatchT](Metric[TaskCompletionMetricResult[Any]]):
    goal_position: Array[Dims[D[2]]]
    optimal_distance: float
    distance_threshold: float
    time_step_size: float
    position_extractor: PositionExtractor[StateBatchT, Positions]

    @staticmethod
    def create(
        *,
        reference: Trajectory[Any, Any],
        distance_threshold: float,
        time_step_size: float,
        position_extractor: PositionExtractor[StateBatchT, Positions],
    ) -> "TaskCompletionMetric[StateBatchT]":
        assert reference.natural_length > 0.0, (
            f"Reference trajectory must have positive length, got {reference.natural_length}."
        )

        return TaskCompletionMetric(
            goal_position=array(reference.end, shape=(2,)),
            optimal_distance=reference.natural_length,
            distance_threshold=distance_threshold,
            time_step_size=time_step_size,
            position_extractor=position_extractor,
        )

    def compute[T: int = int](
        self, data: SimulationData
    ) -> TaskCompletionMetricResult[T]:
        states = data(
            access.states.assume(StateSequence[T, Any, StateBatchT]).require()
        )
        positions = np.asarray(self.position_extractor(states.batched()))[..., 0]

        distances_from_goal = np.linalg.norm(positions - self.goal_position, axis=1)
        completion = distances_from_goal <= self.distance_threshold
        traversed_distance = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()

        return TaskCompletionMetricResult(
            completion=completion,
            completion_time=(
                float(np.argmax(completion)) * self.time_step_size
                if completion.any()
                else float("inf")
            ),
            stretch=float(traversed_distance / self.optimal_distance),
        )

    @property
    def name(self) -> str:
        return "task-completion"
