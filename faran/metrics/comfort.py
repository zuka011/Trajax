from typing import Any
from dataclasses import dataclass

from faran.types import (
    StateSequence,
    SimulationData,
    Metric,
    Positions,
    PositionExtractor,
    Trajectory,
    LateralPositions,
)
from faran.collectors import access

from numtypes import Array, Dims, shape_of

import numpy as np


@dataclass(kw_only=True, frozen=True)
class ComfortMetricResult[T: int = int]:
    """Results of the comfort metric, including lateral acceleration and jerk."""

    lateral_acceleration: Array[Dims[T]]
    lateral_jerk: Array[Dims[T]]


@dataclass(kw_only=True, frozen=True)
class ComfortMetric[StateBatchT, PositionsT, LateralT](
    Metric[ComfortMetricResult[Any]]
):
    """Metric evaluating lateral acceleration and jerk relative to a reference trajectory."""

    reference: Trajectory[Any, Any, PositionsT, LateralT]
    time_step_size: float
    position_extractor: PositionExtractor[StateBatchT, PositionsT]

    @staticmethod
    def create[S, P: Positions, L: LateralPositions](
        *,
        reference: Trajectory[Any, Any, P, L],
        time_step_size: float,
        position_extractor: PositionExtractor[S, P],
    ) -> "ComfortMetric[S, P, L]":
        assert time_step_size > 0, (
            f"Time step size must be positive, got {time_step_size}"
        )

        return ComfortMetric(
            reference=reference,
            time_step_size=time_step_size,
            position_extractor=position_extractor,
        )

    def compute[T: int = int](self, data: SimulationData) -> ComfortMetricResult[Any]:
        states = data(
            access.states.assume(StateSequence[T, Any, StateBatchT]).require()
        )
        state_batch = states.batched()

        positions = self.position_extractor(state_batch)
        lateral = self.reference.lateral(positions)

        lateral_array = np.asarray(lateral)[:, 0]
        lateral_velocity = checked_gradient(lateral_array, self.time_step_size)
        lateral_acceleration = checked_gradient(lateral_velocity, self.time_step_size)
        lateral_jerk = checked_gradient(lateral_acceleration, self.time_step_size)

        return ComfortMetricResult(
            lateral_acceleration=lateral_acceleration, lateral_jerk=lateral_jerk
        )

    @property
    def name(self) -> str:
        return "comfort"


def checked_gradient[L: int](y: Array[Dims[L]], dx: float) -> Array[Dims[L]]:
    if (points := y.shape[0]) < 2:
        return np.zeros_like(y)

    gradient = np.gradient(y, dx)

    assert shape_of(gradient, matches=(points,), name="gradient")

    return gradient
