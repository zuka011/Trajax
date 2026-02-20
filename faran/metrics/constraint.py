from typing import Any
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    StateSequence,
    SimulationData,
    Metric,
    Positions,
    LateralPositions,
    PositionExtractor,
    Trajectory,
    BoundaryDistance,
    BoundaryDistanceExtractor,
)
from faran.collectors import access

from numtypes import Array, BoolArray, Dims

import numpy as np


@dataclass(kw_only=True, frozen=True)
class ConstraintViolationMetricResult[T: int = int]:
    """Results of the constraint violation metric, including boundary distances."""

    lateral_deviations: Array[Dims[T]]
    boundary_distances: Array[Dims[T]]

    @cached_property
    def violations(self) -> BoolArray[Dims[T]]:
        return self.boundary_distances <= 0

    @cached_property
    def violation_detected(self) -> bool:
        return bool(self.violations.any())


@dataclass(kw_only=True, frozen=True)
class ConstraintViolationMetric[StateBatchT, PositionsT, LateralT, BoundaryDistanceT](
    Metric[ConstraintViolationMetricResult[Any]]
):
    """Metric evaluating lateral deviations and boundary constraint violations."""

    reference: Trajectory[Any, Any, PositionsT, LateralT]
    boundary: BoundaryDistanceExtractor[StateBatchT, BoundaryDistanceT]
    position_extractor: PositionExtractor[StateBatchT, PositionsT]

    @staticmethod
    def create[S, P: Positions, L: LateralPositions, BD: BoundaryDistance](
        *,
        reference: Trajectory[Any, Any, P, L],
        boundary: BoundaryDistanceExtractor[S, BD],
        position_extractor: PositionExtractor[S, P],
    ) -> "ConstraintViolationMetric[S, P, L, BD]":
        return ConstraintViolationMetric(
            reference=reference,
            boundary=boundary,
            position_extractor=position_extractor,
        )

    def compute[T: int = int](
        self, data: SimulationData
    ) -> ConstraintViolationMetricResult[T]:
        states = data(
            access.states.assume(StateSequence[T, Any, StateBatchT]).require()
        )
        state_batch = states.batched()

        positions = self.position_extractor(state_batch)
        lateral = self.reference.lateral(positions)
        boundary_distance = self.boundary(states=state_batch)

        lateral_array = np.asarray(lateral)[:, 0]
        boundary_array = np.asarray(boundary_distance)[:, 0]

        return ConstraintViolationMetricResult(
            lateral_deviations=lateral_array, boundary_distances=boundary_array
        )

    @property
    def name(self) -> str:
        return "constraint-violation"
