from typing import Protocol, Final

from faran.types.array import DataType

from numtypes import Array, Dims, D

POSE_D_O: Final = 3

type PoseD_o = D[3]
"""Dimension of a single obstacle pose state (x, y, heading)."""


class SampledObstacleStates[T: int, D_o: int, K: int, N: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        """Returns the sampled states of obstacles as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """The time horizon over which the obstacle states are defined."""
        ...

    @property
    def dimension(self) -> int:
        """The dimension of a single obstacle state."""
        ...

    @property
    def count(self) -> int:
        """The number of obstacles."""
        ...

    @property
    def sample_count(self) -> int:
        """The number of samples per obstacle."""
        ...


class SampledObstaclePositions[T: int, K: int, N: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[2], K, N]]:
        """Returns the sampled positions of obstacles as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, K, N]]:
        """Returns the x positions of obstacles over time and samples."""
        ...

    def y(self) -> Array[Dims[T, K, N]]:
        """Returns the y positions of obstacles over time and samples."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon over which the positions are defined."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...

    @property
    def sample_count(self) -> N:
        """The number of samples per obstacle."""
        ...


class SampledObstacleHeadings[T: int, K: int, N: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, K, N]]:
        """Returns the sampled headings of obstacles as a NumPy array."""
        ...

    def heading(self) -> Array[Dims[T, K, N]]:
        """Returns the headings of obstacles over time and samples."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon over which the headings are defined."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...

    @property
    def sample_count(self) -> N:
        """The number of samples per obstacle."""
        ...


class ObstacleStatesForTimeStep[D_o: int, K: int, ObstacleStatesT](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_o, K]]:
        """Returns the mean states of obstacles at a single time step as a NumPy array."""
        ...

    def replicate(self, *, horizon: int) -> ObstacleStatesT:
        """Replicates the obstacle states over the given time horizon."""
        ...

    @property
    def dimension(self) -> D_o:
        """The dimension of a single obstacle state."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class ObstacleStates[T: int, D_o: int, K: int, SingleSampleT](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        """Returns the mean states of obstacles as a NumPy array."""
        ...

    def covariance(self) -> Array[Dims[T, D_o, D_o, K]] | None:
        """Returns the covariance matrices for the obstacle states x, y, heading over time, if
        one exists."""
        ...

    def single(self) -> SingleSampleT:
        """Returns the single (mean) sample of the obstacle states."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon over which the obstacle states are defined."""
        ...

    @property
    def dimension(self) -> D_o:
        """The dimension of a single obstacle state."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class ObstacleStateProvider[ObstacleStatesT](Protocol):
    def __call__(self) -> ObstacleStatesT:
        """Provides a forecast of states over the time horizon for each obstacle."""
        ...


class ObstacleStateSampler[ObstacleStatesT, SampledObstacleStatesT](Protocol):
    def __call__(
        self, states: ObstacleStatesT, *, count: int
    ) -> SampledObstacleStatesT:
        """Samples the obstacle state forecasts."""
        ...


class Distance[T: int, V: int, M: int, N: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, V, M, N]]:
        """Returns the distances between ego parts and obstacles as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon over which the distances are computed."""
        ...

    @property
    def vehicle_parts(self) -> V:
        """The number of ego vehicle parts for which distances are computed."""
        ...

    @property
    def rollout_count(self) -> M:
        """The number of rollouts for which distances are computed."""
        ...

    @property
    def sample_count(self) -> N:
        """The number of obstacle samples for which distances are computed."""
        ...


class SampledObstaclePositionExtractor[SampledStatesT, PositionsT](Protocol):
    def __call__(self, states: SampledStatesT, /) -> PositionsT:
        """Extracts the positions from the given sampled obstacle states."""
        ...


class SampledObstacleHeadingExtractor[SampledStatesT, HeadingsT](Protocol):
    def __call__(self, states: SampledStatesT, /) -> HeadingsT:
        """Extracts the headings from the given sampled obstacle states."""
        ...


class DistanceExtractor[StateBatchT, SampledObstacleStatesT, DistanceT](Protocol):
    def __call__(
        self, *, states: StateBatchT, obstacle_states: SampledObstacleStatesT
    ) -> DistanceT:
        """Computes the distances between each part of the ego and the corresponding closest
        obstacles."""
        ...


class SampleCostFunction[StateBatchT, SampledObstacleStatesT, CostsT](Protocol):
    def __call__(
        self, *, states: StateBatchT, samples: SampledObstacleStatesT
    ) -> CostsT:
        """Computes the cost given the states and sampled obstacle states."""
        ...


class Risk[T: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        """Returns the risk values as a NumPy array."""
        ...

    @property
    def horizon(self) -> int:
        """The time horizon over which the risk is computed."""
        ...

    @property
    def rollout_count(self) -> int:
        """The number of rollouts for which the risk is computed."""
        ...


class RiskMetric[CostFunctionT, StateBatchT, ObstacleStatesT, SamplerT, RiskT](
    Protocol
):
    def compute(
        self,
        cost_function: CostFunctionT,
        *,
        states: StateBatchT,
        obstacle_states: ObstacleStatesT,
        sampler: SamplerT,
    ) -> RiskT:
        """Computes the risk metric based on the provided cost function."""
        ...

    @property
    def name(self) -> str:
        """Returns a human-readable name for the risk metric."""
        ...
