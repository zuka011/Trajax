from typing import Any
from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    StateSequence,
    SimulationData,
    Metric,
    Positions,
    LongitudinalPositions,
    PositionExtractor,
    Trajectory,
)
from trajax.collectors import access

from numtypes import BoolArray, Array, Dims, D, array

import numpy as np


@dataclass(kw_only=True, frozen=True)
class TaskCompletionMetricResult[T: int = int]:
    """Results of the task completion metric.

    Attributes:
        completion: Boolean array indicating whether the task was completed at each time step.
        completion_time: Time at which the task was first completed, or inf if not completed.
        stretch: Ratio of traversed distance to optimal distance.
        completed_part: Proportion of the reference trajectory completed, including lap count.
            Progress within the current lap is clamped to [0, 1].
            Laps are counted when longitudinal position wraps around (drops by >50% of path length).
    """

    completion: BoolArray[Dims[T]]
    completion_time: float
    stretch: float
    completed_part: float

    @cached_property
    def completed(self) -> bool:
        return bool(self.completion.any())


@dataclass(kw_only=True, frozen=True)
class TaskCompletionMetric[
    StateBatchT,
    PositionsT: Positions,
    LongitudinalT: LongitudinalPositions,
](Metric[TaskCompletionMetricResult[Any]]):
    reference: Trajectory[Any, Any, PositionsT, Any, LongitudinalT]
    goal_position: Array[Dims[D[2]]]
    optimal_distance: float
    path_length: float
    distance_threshold: float
    lap_detection_threshold: float
    time_step_size: float
    position_extractor: PositionExtractor[StateBatchT, PositionsT]

    @staticmethod
    def create[S, P: Positions, L: LongitudinalPositions](
        *,
        reference: Trajectory[Any, Any, P, Any, L],
        distance_threshold: float,
        time_step_size: float,
        position_extractor: PositionExtractor[S, P],
        lap_detection_threshold: float = 0.5,
    ) -> "TaskCompletionMetric[S, P, L]":
        assert reference.natural_length > 0.0, (
            f"Reference trajectory must have positive length, got {reference.natural_length}."
        )

        return TaskCompletionMetric(
            reference=reference,
            goal_position=array(reference.end, shape=(2,)),
            optimal_distance=reference.natural_length,
            path_length=reference.path_length,
            distance_threshold=distance_threshold,
            time_step_size=time_step_size,
            position_extractor=position_extractor,
            lap_detection_threshold=lap_detection_threshold,
        )

    def compute[T: int = int](
        self, data: SimulationData
    ) -> TaskCompletionMetricResult[T]:
        states = data(
            access.states.assume(StateSequence[T, Any, StateBatchT]).require()
        )
        positions = self.position_extractor(states.batched())
        positions_array = np.asarray(positions)[..., 0]

        return TaskCompletionMetricResult(
            completion=(completion := self._compute_completion(positions_array)),
            completion_time=self._compute_completion_time(completion),
            stretch=self._compute_stretch(positions_array),
            completed_part=self._compute_completed_part(positions),
        )

    @property
    def name(self) -> str:
        return "task-completion"

    def _compute_completion[T: int](
        self, positions: Array[Dims[T, D[2]]]
    ) -> BoolArray[Dims[T]]:
        distances_from_goal = np.linalg.norm(positions - self.goal_position, axis=1)
        return distances_from_goal <= self.distance_threshold

    def _compute_completion_time(self, completion: BoolArray[Dims[int]]) -> float:
        return (
            float(np.argmax(completion)) * self.time_step_size
            if completion.any()
            else float("inf")
        )

    def _compute_stretch(self, positions: Array[Dims[int, D[2]]]) -> float:
        traversed_distance = np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()

        return float(traversed_distance / self.optimal_distance)

    def _compute_completed_part(self, positions: PositionsT) -> float:
        longitudinal = np.asarray(self.reference.longitudinal(positions))[:, 0]
        normalized = longitudinal / self.path_length
        clamped = np.clip(normalized, 0.0, 1.0)

        if len(normalized) < 2:
            return float(clamped[-1])

        differences = np.diff(normalized)
        wrap_arounds = differences < -self.lap_detection_threshold
        lap_count = int(np.sum(wrap_arounds))

        return float(lap_count + clamped[-1])
