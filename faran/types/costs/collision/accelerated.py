from typing import Protocol, Any, cast
from dataclasses import dataclass
from functools import cached_property

from faran.types.array import jaxtyped, DataType
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
from jaxtyping import Float, Array as JaxArray

import numpy as np


class JaxSampledObstacleStates[T: int, D_o: int, K: int, N: int](
    SampledObstacleStates[T, D_o, K, N], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_o K N"]:
        """Returns the sampled states of obstacles as a JAX array."""
        ...


class JaxObstacleStatesForTimeStep[D_o: int, K: int, ObstacleStatesT, NumPyT = Any](
    ObstacleStatesForTimeStep[D_o, K, ObstacleStatesT], Protocol
):
    def numpy(self) -> NumPyT:
        """Returns the states of obstacles at a specific time step wrapped for the NumPy backend."""
        ...

    @property
    def array(self) -> Float[JaxArray, "D_o K"]:
        """Returns the states of obstacles at a specific time step as a JAX array."""
        ...


class JaxObstacleStates[
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
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the states of obstacles as a JAX array."""
        ...

    @property
    def covariance_array(self) -> Float[JaxArray, "T D_o D_o K"] | None:
        """Returns the covariances of obstacles as a JAX array, or None if not available."""
        ...


@jaxtyped
@dataclass(frozen=True)
class JaxSampledObstaclePositions[T: int, K: int, N: int](
    SampledObstaclePositions[T, K, N]
):
    _x: Float[JaxArray, "T K N"]
    _y: Float[JaxArray, "T K N"]

    @staticmethod
    def create[T_: int, K_: int, N_: int](
        *,
        x: Float[JaxArray, "T K N"],
        y: Float[JaxArray, "T K N"],
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
        sample_count: N_ | None = None,
    ) -> "JaxSampledObstaclePositions[T_, K_, N_]":
        horizon = horizon if horizon is not None else cast(T_, x.shape[0])
        obstacle_count = (
            obstacle_count if obstacle_count is not None else cast(K_, x.shape[1])
        )
        sample_count = (
            sample_count if sample_count is not None else cast(N_, x.shape[2])
        )

        assert x.shape == y.shape == (horizon, obstacle_count, sample_count), (
            f"Expected shape (T={horizon}, K={obstacle_count}, N={sample_count}), "
            f"but got x: {x.shape}, y: {y.shape}."
        )

        return JaxSampledObstaclePositions(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[2], K, N]]:
        return self._numpy_array

    def x(self) -> Array[Dims[T, K, N]]:
        return self._numpy_x

    def y(self) -> Array[Dims[T, K, N]]:
        return self._numpy_y

    @property
    def horizon(self) -> T:
        return cast(T, self._x.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[1])

    @property
    def sample_count(self) -> N:
        return cast(N, self._x.shape[2])

    @property
    def x_array(self) -> Float[JaxArray, "T K N"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "T K N"]:
        return self._y

    @cached_property
    def _numpy_x(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._x)

    @cached_property
    def _numpy_y(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._y)

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D[2], K, N]]:
        return np.stack([self._numpy_x, self._numpy_y], axis=1)


@jaxtyped
@dataclass(frozen=True)
class JaxSampledObstacleHeadings[T: int, K: int, N: int](
    SampledObstacleHeadings[T, K, N]
):
    _heading: Float[JaxArray, "T K N"]

    @staticmethod
    def create[T_: int, K_: int, N_: int](
        *,
        heading: Float[JaxArray, "T K N"],
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
        sample_count: N_ | None = None,
    ) -> "JaxSampledObstacleHeadings[T_, K_, N_]":
        horizon = horizon if horizon is not None else cast(T_, heading.shape[0])
        obstacle_count = (
            obstacle_count if obstacle_count is not None else cast(K_, heading.shape[1])
        )
        sample_count = (
            sample_count if sample_count is not None else cast(N_, heading.shape[2])
        )

        assert heading.shape == (horizon, obstacle_count, sample_count), (
            f"Expected shape (T={horizon}, K={obstacle_count}, N={sample_count}), "
            f"but got {heading.shape}."
        )

        return JaxSampledObstacleHeadings(_heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, K, N]]:
        return self._numpy_heading

    def heading(self) -> Array[Dims[T, K, N]]:
        return self._numpy_heading

    @property
    def horizon(self) -> T:
        return cast(T, self._heading.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self._heading.shape[1])

    @property
    def sample_count(self) -> N:
        return cast(N, self._heading.shape[2])

    @property
    def heading_array(self) -> Float[JaxArray, "T K N"]:
        return self._heading

    @cached_property
    def _numpy_heading(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._heading)


class JaxObstacleStateProvider[ObstacleStatesT](
    ObstacleStateProvider[ObstacleStatesT], Protocol
): ...


class JaxObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT](
    ObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT], Protocol
): ...


class JaxSampledObstaclePositionExtractor[SampledStatesT](
    SampledObstaclePositionExtractor[SampledStatesT, JaxSampledObstaclePositions],
    Protocol,
): ...


class JaxSampledObstacleHeadingExtractor[SampledStatesT](
    SampledObstacleHeadingExtractor[SampledStatesT, JaxSampledObstacleHeadings],
    Protocol,
): ...


class JaxDistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT](
    DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT], Protocol
): ...


@jaxtyped
@dataclass(frozen=True)
class JaxRisk[T: int, M: int](Risk[T, M]):
    _array: Float[JaxArray, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return np.asarray(self.array)

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[1])

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        return self._array


class JaxRiskMetric[StateBatchT, ObstacleStatesT, SampledObstacleStatesT](
    RiskMetric, Protocol
):
    def compute(
        self,
        cost_function: SampleCostFunction[
            StateBatchT, SampledObstacleStatesT, Float[JaxArray, "T M N"]
        ],
        *,
        states: StateBatchT,
        obstacle_states: ObstacleStatesT,
        sampler: JaxObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT],
    ) -> JaxRisk:
        """Computes the risk metric based on the provided cost function and returns it as a JAX array."""
        ...
