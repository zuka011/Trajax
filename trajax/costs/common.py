from typing import Protocol

from trajax.type import DataType
from trajax.mppi import StateBatch, ControlInputBatch, Costs, CostFunction

from numtypes import Array, Dims


class Error[T: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        """Returns the error as a NumPy array."""
        ...


class ContouringCost[
    InputT: ControlInputBatch,
    StateT: StateBatch,
    CostT: Costs = Costs,
    ErrorT: Error = Error,
](CostFunction[InputT, StateT, CostT], Protocol):
    def error(self, *, inputs: InputT, states: StateT) -> ErrorT:
        """Computes the contouring error for the given inputs and states."""
        ...


class ObstacleStates[T: int, D_o: int, K: int](Protocol):
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


class ObstacleStateProvider[StateT: ObstacleStates](Protocol):
    def __call__(self) -> StateT:
        """Provides the current obstacle states."""
        ...


class Distance[T: int, V: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, V, M]]:
        """Returns the distances between ego parts and obstacles as a NumPy array."""
        ...


class DistanceExtractor[
    StateT: StateBatch,
    ObstacleStatesT: ObstacleStates,
    DistanceT: Distance,
](Protocol):
    def __call__(
        self, *, states: StateT, obstacle_states: ObstacleStatesT
    ) -> DistanceT:
        """Computes the distances between each part of the ego and the corresponding closest
        obstacles."""
        ...


class CollisionCost[
    InputT: ControlInputBatch,
    StateT: StateBatch,
    CostT: Costs = Costs,
](CostFunction[InputT, StateT, CostT]): ...
