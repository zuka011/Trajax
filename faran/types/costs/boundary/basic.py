from typing import Protocol
from dataclasses import dataclass

from faran.types.array import DataType
from faran.types.costs.boundary.common import BoundaryDistanceExtractor

from numtypes import Array, Dims


class NumPyBoundaryDistanceExtractor[StateBatchT, DistanceT](
    BoundaryDistanceExtractor[StateBatchT, DistanceT], Protocol
): ...


@dataclass(frozen=True)
class NumPyBoundaryDistance[T: int, M: int]:
    _array: Array[Dims[T, M]]

    @staticmethod
    def create[T_: int, M_: int](
        *, array: Array[Dims[T_, M_]]
    ) -> "NumPyBoundaryDistance[T_, M_]":
        """Creates a NumPy boundary distance from the given array."""
        return NumPyBoundaryDistance(array)

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
