from typing import Protocol, cast
from dataclasses import dataclass
from functools import cached_property

from trajax.types.array import DataType, jaxtyped
from trajax.types.costs.boundary.common import BoundaryDistanceExtractor

from numtypes import Array, Dims
from jaxtyping import Array as JaxArray, Float

import numpy as np


class JaxBoundaryDistanceExtractor[StateBatchT, DistanceT](
    BoundaryDistanceExtractor[StateBatchT, DistanceT], Protocol
): ...


@jaxtyped
@dataclass(frozen=True)
class JaxBoundaryDistance[T: int, M: int]:
    _array: Float[JaxArray, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return self._numpy_array

    @property
    def horizon(self) -> T:
        return cast(T, self._array.shape[0])

    @property
    def rollout_count(self) -> M:
        return cast(M, self._array.shape[1])

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, M]]:
        return np.array(self._array)
