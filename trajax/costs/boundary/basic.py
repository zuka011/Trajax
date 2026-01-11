from dataclasses import dataclass

from trajax.types import (
    CostFunction,
    ControlInputBatch,
    NumPyBoundaryDistance,
    NumPyBoundaryDistanceExtractor,
    NumPyCosts,
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
    ) -> "NumPyBoundaryCost":
        return NumPyBoundaryCost(
            distance=distance, distance_threshold=distance_threshold, weight=weight
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> NumPyCosts[T, M]:
        cost = (
            np.asarray(self.distance_threshold)[np.newaxis, np.newaxis]
            - self.distance(states=states).array
        )

        return NumPySimpleCosts(self.weight * np.clip(cost, 0, None))
