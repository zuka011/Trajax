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
    CostT: Costs,
    ErrorT: Error,
](CostFunction[InputT, StateT, CostT], Protocol):
    def error(self, *, inputs: InputT, states: StateT) -> ErrorT:
        """Computes the contouring error for the given inputs and states."""
        ...
