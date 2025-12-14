from typing import Protocol
from dataclasses import dataclass

from trajax.types import types
from trajax.model import ControlInputBatch, StateBatch
from trajax.trajectory import Trajectory
from trajax.mppi import Costs

import numpy as np
from numtypes import Array, Dim2


type PathParameters[T: int = int, M: int = int] = types.numpy.PathParameters[T, M]
type Positions[T: int = int, M: int = int] = types.numpy.Positions[T, M]
type ReferencePoints[T: int = int, M: int = int] = types.numpy.ReferencePoints[T, M]


class PathParameterExtractor[StateT: StateBatch](Protocol):
    def __call__(self, states: StateT, /) -> PathParameters:
        """Extracts path parameters from a batch of states."""
        ...


class PathVelocityExtractor[InputT: ControlInputBatch](Protocol):
    def __call__(self, inputs: InputT, /) -> Array[Dim2]:
        """Extracts path velocities from a batch of control inputs."""
        ...


class PositionExtractor[StateT: StateBatch](Protocol):
    def __call__(self, states: StateT, /) -> Positions:
        """Extracts (x, y) positions from a batch of states."""
        ...


@dataclass(kw_only=True, frozen=True)
class ContouringCost[StateT: StateBatch]:
    reference: Trajectory[PathParameters, ReferencePoints]
    path_parameter_extractor: PathParameterExtractor[StateT]
    position_extractor: PositionExtractor[StateT]
    weight: float

    @staticmethod
    def create[S: StateBatch](
        *,
        reference: Trajectory[PathParameters, ReferencePoints],
        path_parameter_extractor: PathParameterExtractor[S],
        position_extractor: PositionExtractor[S],
        weight: float,
    ) -> "ContouringCost[S]":
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

    def __call__(self, *, inputs: ControlInputBatch, states: StateT) -> Costs:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading
        positions = self.position_extractor(states)

        error = np.sin(heading) * (positions.x - ref_points.x) - np.cos(heading) * (
            positions.y - ref_points.y
        )

        return types.numpy.basic.costs(self.weight * error**2)


@dataclass(kw_only=True, frozen=True)
class LagCost[StateT: StateBatch]:
    reference: Trajectory[PathParameters, ReferencePoints]
    path_parameter_extractor: PathParameterExtractor[StateT]
    position_extractor: PositionExtractor[StateT]
    weight: float

    @staticmethod
    def create[S: StateBatch](
        *,
        reference: Trajectory[PathParameters, ReferencePoints],
        path_parameter_extractor: PathParameterExtractor[S],
        position_extractor: PositionExtractor[S],
        weight: float,
    ) -> "LagCost[S]":
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
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> Costs[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading
        positions = self.position_extractor(states)

        error = np.cos(heading) * (positions.x - ref_points.x) + np.sin(heading) * (
            positions.y - ref_points.y
        )

        return types.numpy.basic.costs(self.weight * error**2)


@dataclass(kw_only=True, frozen=True)
class ProgressCost[InputT: ControlInputBatch]:
    path_velocity_extractor: PathVelocityExtractor[InputT]
    time_step_size: float
    weight: float

    @staticmethod
    def create[I: ControlInputBatch](
        *,
        path_velocity_extractor: PathVelocityExtractor[I],
        time_step_size: float,
        weight: float,
    ) -> "ProgressCost[I]":
        """Creates a progress cost implemented with NumPy.

        Args:
            path_velocity_extractor: Extracts the path velocities from control input batch.
            time_step_size: The time step size between states.
            weight: The weight of the progress cost.
        """
        return ProgressCost(
            path_velocity_extractor=path_velocity_extractor,
            time_step_size=time_step_size,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: InputT, states: StateBatch[T, int, M]
    ) -> Costs[T, M]:
        path_velocities = self.path_velocity_extractor(inputs)

        return types.numpy.basic.costs(
            -self.weight * path_velocities * self.time_step_size
        )
