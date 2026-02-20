from typing import Protocol, Any
from dataclasses import dataclass
from functools import cached_property

from faran.types.array import DataType
from faran.types.costs.collision.common import (
    ObstacleStateProvider,
    ObstacleStateSampler,
    SampledObstacleStates,
    SampledObstaclePositions,
    SampledObstacleHeadings,
    SampledObstaclePositionExtractor,
    SampledObstacleHeadingExtractor,
    ObstacleStatesForTimeStep,
    ObstacleStates,
    DistanceExtractor,
    SampleCostFunction,
    Risk,
    RiskMetric,
)

from numtypes import Array, Dims, D

import numpy as np


class NumPySampledObstacleStates[T: int, D_o: int, K: int, N: int](
    SampledObstacleStates[T, D_o, K, N], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_o, K, N]]:
        """Returns the sampled states of obstacles as a NumPy array."""
        ...


class NumPyObstacleStatesForTimeStep[D_o: int, K: int, ObstacleStatesT](
    ObstacleStatesForTimeStep[D_o, K, ObstacleStatesT], Protocol
):
    @property
    def array(self) -> Array[Dims[D_o, K]]:
        """Returns the states of obstacles at a specific time step as a NumPy array."""
        ...


class NumPyObstacleStates[
    T: int,
    D_o: int,
    K: int,
    SingleSampleT,
    ObstacleStatesForTimeStepT = Any,
](ObstacleStates[T, D_o, K, SingleSampleT], Protocol):
    def last(self) -> ObstacleStatesForTimeStepT:
        """Returns the states of obstacles at the last time step."""
        ...

    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
        """Returns the states of obstacles as a NumPy array."""
        ...


@dataclass(frozen=True)
class NumPySampledObstaclePositions[T: int, K: int, N: int](
    SampledObstaclePositions[T, K, N]
):
    _x: Array[Dims[T, K, N]]
    _y: Array[Dims[T, K, N]]

    @staticmethod
    def create[T_: int, K_: int, N_: int](
        *, x: Array[Dims[T_, K_, N_]], y: Array[Dims[T_, K_, N_]]
    ) -> "NumPySampledObstaclePositions[T_, K_, N_]":
        return NumPySampledObstaclePositions(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[2], K, N]]:
        return self.array

    def x(self) -> Array[Dims[T, K, N]]:
        return self._x

    def y(self) -> Array[Dims[T, K, N]]:
        return self._y

    @property
    def horizon(self) -> T:
        return self._x.shape[0]

    @property
    def count(self) -> K:
        return self._x.shape[1]

    @property
    def sample_count(self) -> N:
        return self._x.shape[2]

    @property
    def array(self) -> Array[Dims[T, D[2], K, N]]:
        return self._array

    @cached_property
    def _array(self) -> Array[Dims[T, D[2], K, N]]:
        return np.stack([self._x, self._y], axis=1)


@dataclass(frozen=True)
class NumPySampledObstacleHeadings[T: int, K: int, N: int](
    SampledObstacleHeadings[T, K, N]
):
    _heading: Array[Dims[T, K, N]]

    @staticmethod
    def create[T_: int, K_: int, N_: int](
        *, heading: Array[Dims[T_, K_, N_]]
    ) -> "NumPySampledObstacleHeadings[T_, K_, N_]":
        return NumPySampledObstacleHeadings(_heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, K, N]]:
        return self.array

    def heading(self) -> Array[Dims[T, K, N]]:
        return self._heading

    @property
    def horizon(self) -> T:
        return self._heading.shape[0]

    @property
    def count(self) -> K:
        return self._heading.shape[1]

    @property
    def sample_count(self) -> N:
        return self._heading.shape[2]

    @property
    def array(self) -> Array[Dims[T, K, N]]:
        return self._heading


class NumPyObstacleStateProvider[ObstacleStatesT](
    ObstacleStateProvider[ObstacleStatesT], Protocol
): ...


class NumPyObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT](
    ObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT], Protocol
): ...


class NumPySampledObstaclePositionExtractor[SampledStatesT](
    SampledObstaclePositionExtractor[SampledStatesT, NumPySampledObstaclePositions],
    Protocol,
): ...


class NumPySampledObstacleHeadingExtractor[SampledStatesT](
    SampledObstacleHeadingExtractor[SampledStatesT, NumPySampledObstacleHeadings],
    Protocol,
): ...


class NumPyDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT](
    DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT], Protocol
): ...


@dataclass(frozen=True)
class NumPyRisk[T: int, M: int](Risk[T, M]):
    _array: Array[Dims[T, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return self._array

    @property
    def horizon(self) -> T:
        return self._array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self._array.shape[1]

    @property
    def array(self) -> Array[Dims[T, M]]:
        return self._array


class NumPyRiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT](
    RiskMetric, Protocol
):
    def compute[T: int, M: int](
        self,
        cost_function: SampleCostFunction[
            StateBatchT, SampledObstacleStatesT, Array[Dims[T, M, int]]
        ],
        *,
        states: StateBatchT,
        obstacle_states: ObstacleStatesT,
        sampler: NumPyObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT],
    ) -> NumPyRisk[T, M]:
        """Computes the risk metric based on the provided cost function and returns it as a NumPy array."""
        ...
