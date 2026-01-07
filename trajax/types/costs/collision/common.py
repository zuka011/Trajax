from typing import Protocol, Final

from trajax.types.array import DataType

from numtypes import Array, Dims, D

D_O: Final = 3

type D_o = D[3]


class SampledObstacleStates[T: int, K: int, N: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        """Returns the sampled states of obstacles as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, K, N]]:
        """Returns the x positions of obstacles over time and samples."""
        ...

    def y(self) -> Array[Dims[T, K, N]]:
        """Returns the y positions of obstacles over time and samples."""
        ...

    def heading(self) -> Array[Dims[T, K, N]]:
        """Returns the headings of obstacles over time and samples."""
        ...


class ObstacleStatesForTimeStep[K: int, ObstacleStatesT](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_o, K]]:
        """Returns the mean states of obstacles at a single time step as a NumPy array."""
        ...

    def x(self) -> Array[Dims[K]]:
        """Returns the x positions of obstacles at a single time step."""
        ...

    def y(self) -> Array[Dims[K]]:
        """Returns the y positions of obstacles at a single time step."""
        ...

    def heading(self) -> Array[Dims[K]]:
        """Returns the headings of obstacles at a single time step."""
        ...

    def replicate(self, *, horizon: int) -> ObstacleStatesT:
        """Replicates the obstacle states over the given time horizon."""
        ...


class ObstacleStates[T: int, K: int, SingleSampleT](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        """Returns the mean states of obstacles as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, K]]:
        """Returns the x positions of obstacles over time."""
        ...

    def y(self) -> Array[Dims[T, K]]:
        """Returns the y positions of obstacles over time."""
        ...

    def heading(self) -> Array[Dims[T, K]]:
        """Returns the headings of obstacles over time."""
        ...

    def covariance(self) -> Array[Dims[T, D_o, D_o, K]] | None:
        """Returns the covariance matrices for the obstacle states x, y, heading over time, if
        one exists."""
        ...

    def single(self) -> SingleSampleT:
        """Returns the single (mean) sample of the obstacle states."""
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
