from typing import Protocol
from dataclasses import dataclass

from trajax.types.array import DataType
from trajax.types.costs.boundary.common import BoundaryDistanceExtractor

from numtypes import Array, Dims


class NumPyBoundaryDistanceExtractor[StateBatchT, DistanceT](
    BoundaryDistanceExtractor[StateBatchT, DistanceT], Protocol
): ...


@dataclass(frozen=True)
class NumPyBoundaryDistance[T: int, M: int]:
    _array: Array[Dims[T, M]]

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
