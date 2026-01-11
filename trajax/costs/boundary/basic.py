from typing import Any
from dataclasses import dataclass

from trajax.types import (
    CostFunction,
    ControlInputBatch,
    Trajectory,
    NumPyBoundaryDistance,
    NumPyBoundaryDistanceExtractor,
    NumPyCosts,
    NumPyPositions,
    NumPyLateralPositions,
    NumPyPositionExtractor,
)
from trajax.states import NumPySimpleCosts

import numpy as np


@dataclass(frozen=True)
class NumPyBoundaryCost[StateT](CostFunction[ControlInputBatch, StateT, NumPyCosts]):
    distance: NumPyBoundaryDistanceExtractor[StateT, NumPyBoundaryDistance]
    distance_threshold: float
    weight: float

    @staticmethod
    def create[S](
        *,
        distance: NumPyBoundaryDistanceExtractor[S, NumPyBoundaryDistance],
        distance_threshold: float,
        weight: float,
    ) -> "NumPyBoundaryCost[S]":
        return NumPyBoundaryCost(
            distance=distance, distance_threshold=distance_threshold, weight=weight
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> NumPyCosts[T, M]:
        cost = self.distance_threshold - self.distance(states=states).array

        return NumPySimpleCosts(self.weight * np.clip(cost, 0, None))


@dataclass(kw_only=True, frozen=True)
class NumPyFixedWidthBoundary[StateT](
    NumPyBoundaryDistanceExtractor[StateT, NumPyBoundaryDistance]
):
    reference: Trajectory[Any, Any, NumPyPositions, NumPyLateralPositions]
    position_extractor: NumPyPositionExtractor[StateT]
    left: float
    right: float

    @staticmethod
    def create[S](
        *,
        reference: Trajectory[Any, Any, NumPyPositions, NumPyLateralPositions],
        position_extractor: NumPyPositionExtractor[S],
        left: float,
        right: float,
    ) -> "NumPyFixedWidthBoundary[S]":
        """Creates a fixed-width boundary distance extractor.

        This component assumes a fixed-width corridor around a reference trajectory. The left and
        right widths can be different (asymmetric corridor).

        Args:
            reference: The reference trajectory defining the center of the corridor.
            position_extractor: Function to extract positions from states.
            left: The width of the left side of the corridor.
            right: The width of the right side of the corridor.
        """
        assert left >= -right, (
            f"The boundaries appear to be inverted. Left: {left}, Right: {right}. "
            f"Make sure the total width (left + right) is non-negative, got {left + right}."
        )

        return NumPyFixedWidthBoundary(
            reference=reference,
            position_extractor=position_extractor,
            left=left,
            right=right,
        )

    def __call__(self, *, states: StateT) -> NumPyBoundaryDistance:
        positions = self.position_extractor(states)
        lateral = self.reference.lateral(positions)

        distance_to_left = self.left + lateral.array
        distance_to_right = self.right - lateral.array

        return NumPyBoundaryDistance(np.minimum(distance_to_left, distance_to_right))
