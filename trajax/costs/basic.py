from typing import Protocol
from dataclasses import dataclass

from trajax.type import DataType
from trajax.mppi import (
    StateBatch,
    ControlInputBatch,
    CostFunction,
    NumPyStateBatch,
    NumPyControlInputBatch,
    NumPyCosts,
)
from trajax.trajectory import (
    Trajectory,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositions,
    NumPyHeadings,
)
from trajax.states import NumPySimpleCosts
from trajax.costs.common import ContouringCost, Error

from numtypes import Array, Dims, Dim2

import numpy as np


class NumPyPathParameterExtractor[StateT: NumPyStateBatch](Protocol):
    def __call__(self, states: StateT, /) -> NumPyPathParameters:
        """Extracts path parameters from a batch of states."""
        ...


class NumPyPathVelocityExtractor[InputT: NumPyControlInputBatch](Protocol):
    def __call__(self, inputs: InputT, /) -> Array[Dim2]:
        """Extracts path velocities from a batch of control inputs."""
        ...


class NumPyPositionExtractor[StateT: NumPyStateBatch](Protocol):
    def __call__(self, states: StateT, /) -> NumPyPositions:
        """Extracts (x, y) positions from a batch of states."""
        ...


class NumPyHeadingExtractor[StateT: NumPyStateBatch](Protocol):
    def __call__(self, states: StateT, /) -> NumPyHeadings:
        """Extracts heading angles from a batch of states."""
        ...


@dataclass(frozen=True)
class NumPyError[T: int, M: int](Error[T, M]):
    array: Array[Dims[T, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return self.array


@dataclass(kw_only=True, frozen=True)
class NumPyContouringCost[StateT: NumPyStateBatch](
    ContouringCost[ControlInputBatch, StateT, NumPyCosts, NumPyError]
):
    reference: Trajectory[NumPyPathParameters, NumPyReferencePoints]
    path_parameter_extractor: NumPyPathParameterExtractor[StateT]
    position_extractor: NumPyPositionExtractor[StateT]
    weight: float

    @staticmethod
    def create[S: NumPyStateBatch](
        *,
        reference: Trajectory[NumPyPathParameters, NumPyReferencePoints],
        path_parameter_extractor: NumPyPathParameterExtractor[S],
        position_extractor: NumPyPositionExtractor[S],
        weight: float,
    ) -> "NumPyContouringCost[S]":
        """Creates a contouring cost implemented with NumPy.

        Args:
            reference: The reference trajectory to follow.
            path_parameter_extractor: Extracts the path parameters from a state batch.
            position_extractor: Extracts the (x, y) positions from a state batch.
            weight: The weight of the contouring cost.
        """
        return NumPyContouringCost(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> NumPySimpleCosts[T, M]:
        error = self.error(inputs=inputs, states=states)
        return NumPySimpleCosts(self.weight * error.array**2)

    def error[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> NumPyError[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading()
        positions = self.position_extractor(states)

        return NumPyError(
            np.sin(heading) * (positions.x - ref_points.x())
            - np.cos(heading) * (positions.y - ref_points.y())
        )


@dataclass(kw_only=True, frozen=True)
class NumPyLagCost[StateT: NumPyStateBatch](
    CostFunction[ControlInputBatch, StateT, NumPyCosts]
):
    reference: Trajectory[NumPyPathParameters, NumPyReferencePoints]
    path_parameter_extractor: NumPyPathParameterExtractor[StateT]
    position_extractor: NumPyPositionExtractor[StateT]
    weight: float

    @staticmethod
    def create[S: NumPyStateBatch](
        *,
        reference: Trajectory[NumPyPathParameters, NumPyReferencePoints],
        path_parameter_extractor: NumPyPathParameterExtractor[S],
        position_extractor: NumPyPositionExtractor[S],
        weight: float,
    ) -> "NumPyLagCost[S]":
        """Creates a lag cost implemented with NumPy.

        Args:
            reference: The reference trajectory to follow.
            path_parameter_extractor: Extracts the path parameters from a state batch.
            position_extractor: Extracts the (x, y) positions from a state batch.
            weight: The weight of the lag cost.
        """
        return NumPyLagCost(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> NumPySimpleCosts[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading()
        positions = self.position_extractor(states)

        error = np.cos(heading) * (positions.x - ref_points.x()) + np.sin(heading) * (
            positions.y - ref_points.y()
        )

        return NumPySimpleCosts(self.weight * error**2)


@dataclass(kw_only=True, frozen=True)
class NumPyProgressCost[InputT: NumPyControlInputBatch](
    CostFunction[InputT, StateBatch, NumPySimpleCosts]
):
    path_velocity_extractor: NumPyPathVelocityExtractor[InputT]
    time_step_size: float
    weight: float

    @staticmethod
    def create[I: NumPyControlInputBatch](
        *,
        path_velocity_extractor: NumPyPathVelocityExtractor[I],
        time_step_size: float,
        weight: float,
    ) -> "NumPyProgressCost[I]":
        """Creates a progress cost implemented with NumPy.

        Args:
            path_velocity_extractor: Extracts the path velocities from a control input batch.
            time_step_size: The time step size between states.
            weight: The weight of the progress cost.
        """
        return NumPyProgressCost(
            path_velocity_extractor=path_velocity_extractor,
            time_step_size=time_step_size,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: InputT, states: StateBatch[T, int, M]
    ) -> NumPySimpleCosts[T, M]:
        path_velocities = self.path_velocity_extractor(inputs)

        return NumPySimpleCosts(-self.weight * path_velocities * self.time_step_size)


@dataclass(kw_only=True, frozen=True)
class NumPyControlSmoothingCost[D_u: int](
    CostFunction[NumPyControlInputBatch[int, D_u, int], StateBatch, NumPySimpleCosts]
):
    weights: Array[Dims[D_u]]

    @staticmethod
    def create[D_u_: int](
        *,
        weights: Array[Dims[D_u_]],
    ) -> "NumPyControlSmoothingCost[D_u_]":
        """Creates a control smoothing cost implemented with NumPy.

        Args:
            weights: The weights for each control input dimension.
        """
        return NumPyControlSmoothingCost(weights=weights)

    def __call__[T: int, M: int](
        self,
        *,
        inputs: NumPyControlInputBatch[T, D_u, M],
        states: StateBatch[T, int, M],
    ) -> NumPySimpleCosts[T, M]:
        input_array = np.asarray(inputs)
        diffs = np.diff(input_array, axis=0, prepend=input_array[0:1, :, :])
        squared_diffs = diffs**2
        weighted_squared_diffs = squared_diffs * self.weights[np.newaxis, :, np.newaxis]
        cost_per_time_step = np.sum(weighted_squared_diffs, axis=1)

        return NumPySimpleCosts(cost_per_time_step)
