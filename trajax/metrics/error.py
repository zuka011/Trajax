from typing import Any
from dataclasses import dataclass

from trajax.types import StateSequence, ContouringCost, LagCost, SimulationData, Metric
from trajax.collectors import access

from numtypes import Array, Dims

import numpy as np


@dataclass(kw_only=True, frozen=True)
class MpccErrorMetricResult[T: int]:
    contouring: Array[Dims[T]]
    lag: Array[Dims[T]]

    @property
    def max_contouring(self) -> float:
        """Returns the maximum absolute contouring error detected."""
        return float(np.abs(self.contouring).max())

    @property
    def max_lag(self) -> float:
        """Returns the maximum absolute lag error detected."""
        return float(np.abs(self.lag).max())


@dataclass(kw_only=True, frozen=True)
class MpccErrorMetric[StateBatchT](Metric[MpccErrorMetricResult]):
    contouring: ContouringCost[Any, StateBatchT]
    lag: LagCost[Any, StateBatchT]

    @staticmethod
    def create(
        *,
        contouring: ContouringCost[Any, StateBatchT],
        lag: LagCost[Any, StateBatchT],
    ) -> "MpccErrorMetric[StateBatchT]":
        return MpccErrorMetric(contouring=contouring, lag=lag)

    def compute[T: int = int](self, data: SimulationData) -> MpccErrorMetricResult[T]:
        states = data(
            access.states.assume(StateSequence[T, Any, StateBatchT]).require()
        )
        state_batch = states.batched()

        contouring_error = self.contouring.error(states=state_batch)
        lag_error = self.lag.error(states=state_batch)

        return MpccErrorMetricResult(
            contouring=np.asarray(contouring_error), lag=np.asarray(lag_error)
        )

    @property
    def name(self) -> str:
        return "mpcc-error"
