from typing import Protocol

from trajax.types.array import DataType

from numtypes import Array, Dims


class Error[T: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        """Returns the error as a NumPy array."""
        ...


class PositionExtractor[StateBatchT, PositionsT](Protocol):
    def __call__(self, states: StateBatchT, /) -> PositionsT:
        """Extracts (x, y) positions from a batch of states."""
        ...


class ContouringCost[InputBatchT, StateBatchT, ErrorT = Error](Protocol):
    def error(self, *, states: StateBatchT) -> ErrorT:
        """Computes the contouring error for the given states."""
        ...


class LagCost[InputBatchT, StateBatchT, ErrorT = Error](Protocol):
    def error(self, *, states: StateBatchT) -> ErrorT:
        """Computes the lag error for the given states."""
        ...
