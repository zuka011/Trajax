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


class Distance[T: int, V: int, M: int]:
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, V, M]]:
        """Returns the distances between ego parts and obstacles as a NumPy array."""
        ...


class DistanceExtractor[StateT: StateBatch, DistanceT: Distance]:
    def __call__(self, state: StateT) -> DistanceT:
        """Computes the distances between each part of the ego and the corresponding closest
        obstacles."""
        ...


class CollisionCost[
    InputT: ControlInputBatch,
    StateT: StateBatch,
    CostT: Costs = Costs,
](CostFunction[InputT, StateT, CostT]): ...
