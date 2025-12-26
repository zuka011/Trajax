from typing import Protocol, Final

from trajax.type import DataType
from trajax.mppi import StateBatch

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


class ObstacleStates[T: int, K: int, SingleSampleT: SampledObstacleStates](Protocol):
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

    def single(self) -> SingleSampleT:
        """Returns the single (mean) sample of the obstacle states."""
        ...


class ObstacleStateProvider[StateT: ObstacleStates](Protocol):
    def __call__(self) -> StateT:
        """Provides the current obstacle states."""
        ...


class ObstacleStateSampler[StateT: ObstacleStates, SampleT: SampledObstacleStates](
    Protocol
):
    def __call__(self, states: StateT, *, count: int) -> SampleT:
        """Samples the provided obstacle states."""
        ...


class Distance[T: int, V: int, M: int, N: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, V, M, N]]:
        """Returns the distances between ego parts and obstacles as a NumPy array."""
        ...


class DistanceExtractor[
    StateT: StateBatch,
    SampleT: SampledObstacleStates,
    DistanceT: Distance,
](Protocol):
    def __call__(self, *, states: StateT, obstacle_states: SampleT) -> DistanceT:
        """Computes the distances between each part of the ego and the corresponding closest
        obstacles."""
        ...
