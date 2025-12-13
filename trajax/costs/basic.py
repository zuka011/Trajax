from typing import Protocol
from dataclasses import dataclass

from trajax.types import types
from trajax.model import ControlInputBatch, StateBatch
from trajax.trajectory import Trajectory
from trajax.mppi import Costs

import numpy as np


type PathParameters[T: int = int, M: int = int] = types.numpy.PathParameters[T, M]
type Positions[T: int = int, M: int = int] = types.numpy.Positions[T, M]
type ReferencePoints[T: int = int, M: int = int] = types.numpy.ReferencePoints[T, M]


class PathParameterExtractor[D_x: int](Protocol):
    def __call__[T: int, M: int](
        self, states: StateBatch[T, D_x, M]
    ) -> PathParameters[T, M]:
        """Extracts path parameters from a batch of states."""
        ...


class PositionExtractor[D_x: int](Protocol):
    def __call__[T: int, M: int](
        self, states: StateBatch[T, D_x, M]
    ) -> Positions[T, M]:
        """Extracts (x, y) positions from a batch of states."""
        ...


@dataclass(kw_only=True, frozen=True)
class ContouringCost[D_x: int]:
    reference: Trajectory[PathParameters, ReferencePoints]
    path_parameter_extractor: PathParameterExtractor[D_x]
    position_extractor: PositionExtractor[D_x]
    weight: float

    @staticmethod
    def create[D_x_: int = int](
        *,
        reference: Trajectory[PathParameters, ReferencePoints],
        path_parameter_extractor: PathParameterExtractor[D_x_],
        position_extractor: PositionExtractor[D_x_],
        weight: float,
    ) -> "ContouringCost[D_x_]":
        """Creates a contouring cost implemented with NumPy.

        Args:
            reference: The reference trajectory to follow.
            path_parameter_extractor: Extracts the path parameters from state batch.
            position_extractor: Extracts the (x, y) positions from a state batch.
            weight: The weight of the contouring cost.
        """
        return ContouringCost(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateBatch[T, D_x, M]
    ) -> Costs[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading
        positions = self.position_extractor(states)

        error = np.sin(heading) * (positions.x - ref_points.x) - np.cos(heading) * (
            positions.y - ref_points.y
        )

        return types.numpy.basic.costs(self.weight * error**2)


@dataclass(kw_only=True, frozen=True)
class LagCost[D_x: int]:
    reference: Trajectory[PathParameters, ReferencePoints]
    path_parameter_extractor: PathParameterExtractor[D_x]
    position_extractor: PositionExtractor[D_x]
    weight: float

    @staticmethod
    def create[D_x_: int = int](
        *,
        reference: Trajectory[PathParameters, ReferencePoints],
        path_parameter_extractor: PathParameterExtractor[D_x_],
        position_extractor: PositionExtractor[D_x_],
        weight: float,
    ) -> "LagCost[D_x_]":
        """Creates a lag cost implemented with NumPy.

        Args:
            reference: The reference trajectory to follow.
            path_parameter_extractor: Extracts the path parameters from state batch.
            position_extractor: Extracts the (x, y) positions from a state batch.
            weight: The weight of the lag cost.
        """
        return LagCost(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateBatch[T, D_x, M]
    ) -> Costs[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading
        positions = self.position_extractor(states)

        error = np.cos(heading) * (positions.x - ref_points.x) + np.sin(heading) * (
            positions.y - ref_points.y
        )

        return types.numpy.basic.costs(self.weight * error**2)
