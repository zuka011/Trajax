from typing import Protocol

from numtypes import Array, Dims


class NumPyControlInputBatchCreator[
    InputBatchT,
    T: int = int,
    D_u: int = int,
    M: int = int,
](Protocol):
    def __call__(self, *, array: Array[Dims[T, D_u, M]]) -> InputBatchT:
        """Creates a control input batch from the given array."""
        ...
