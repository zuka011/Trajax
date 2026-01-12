from dataclasses import dataclass

from trajax.types import (
    DataType,
    StateBatch,
    ControlInputBatch,
    CostFunction,
    ContouringCost,
    LagCost,
    Error,
    Trajectory,
    NumPyControlInputBatch,
    NumPyCosts,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositionExtractor,
    NumPyPathParameterExtractor,
    NumPyPathVelocityExtractor,
)
from trajax.states import NumPySimpleCosts

from numtypes import Array, Dims

import numpy as np


@dataclass(frozen=True)
class NumPyError[T: int, M: int](Error[T, M]):
    array: Array[Dims[T, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return self.array


@dataclass(kw_only=True, frozen=True)
class NumPyContouringCost[StateBatchT](
    ContouringCost[ControlInputBatch, StateBatchT, NumPyError],
    CostFunction[ControlInputBatch, StateBatchT, NumPyCosts],
):
    reference: Trajectory[NumPyPathParameters, NumPyReferencePoints]
    path_parameter_extractor: NumPyPathParameterExtractor[StateBatchT]
    position_extractor: NumPyPositionExtractor[StateBatchT]
    weight: float

    @staticmethod
    def create[S](
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
        self, *, inputs: ControlInputBatch[T, int, M], states: StateBatchT
    ) -> NumPyCosts[T, M]:
        error = self.error(states=states)
        return NumPySimpleCosts(self.weight * error.array**2)

    def error[T: int = int, M: int = int](
        self, *, states: StateBatchT
    ) -> NumPyError[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading()
        positions = self.position_extractor(states)

        return NumPyError(
            np.sin(heading) * (positions.x() - ref_points.x())
            - np.cos(heading) * (positions.y() - ref_points.y())
        )


@dataclass(kw_only=True, frozen=True)
class NumPyLagCost[StateBatchT](
    LagCost[ControlInputBatch, StateBatchT, NumPyError],
    CostFunction[ControlInputBatch, StateBatchT, NumPyCosts],
):
    reference: Trajectory[NumPyPathParameters, NumPyReferencePoints]
    path_parameter_extractor: NumPyPathParameterExtractor[StateBatchT]
    position_extractor: NumPyPositionExtractor[StateBatchT]
    weight: float

    @staticmethod
    def create[S](
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
        self, *, inputs: ControlInputBatch[T, int, M], states: StateBatchT
    ) -> NumPyCosts[T, M]:
        error = self.error(states=states)
        return NumPySimpleCosts(self.weight * error.array**2)

    def error[T: int = int, M: int = int](
        self, *, states: StateBatchT
    ) -> NumPyError[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading()
        positions = self.position_extractor(states)

        return NumPyError(
            -np.cos(heading) * (positions.x() - ref_points.x())
            - np.sin(heading) * (positions.y() - ref_points.y())
        )


@dataclass(kw_only=True, frozen=True)
class NumPyProgressCost[InputBatchT](CostFunction[InputBatchT, StateBatch, NumPyCosts]):
    path_velocity_extractor: NumPyPathVelocityExtractor[InputBatchT]
    time_step_size: float
    weight: float

    @staticmethod
    def create[I](
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
        self, *, inputs: InputBatchT, states: StateBatch[T, int, M]
    ) -> NumPyCosts[T, M]:
        path_velocities = self.path_velocity_extractor(inputs)

        return NumPySimpleCosts(-self.weight * path_velocities * self.time_step_size)


@dataclass(kw_only=True, frozen=True)
class NumPyControlSmoothingCost[D_u: int](
    CostFunction[NumPyControlInputBatch[int, D_u, int], StateBatch, NumPyCosts]
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
    ) -> NumPyCosts[T, M]:
        diffs = np.diff(inputs.array, axis=0, prepend=inputs.array[0:1, :, :])
        squared_diffs = diffs**2
        weighted_squared_diffs = squared_diffs * self.weights[np.newaxis, :, np.newaxis]
        cost_per_time_step = np.sum(weighted_squared_diffs, axis=1)

        return NumPySimpleCosts(cost_per_time_step)
