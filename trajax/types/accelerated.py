from dataclasses import dataclass
from typing import cast

from trajax.type import DataType, jaxtyped

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dims
import numpy as np


@jaxtyped
@dataclass(frozen=True)
class BasicState[D_x: int]:
    array: Float[JaxArray, "D_x"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return np.asarray(self.array, dtype=dtype)


@jaxtyped
@dataclass(frozen=True)
class BasicStateBatch[T: int, D_x: int, M: int]:
    array: Float[JaxArray, "T D_x M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        return np.asarray(self.array, dtype=dtype)

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])


@jaxtyped
@dataclass(frozen=True)
class BasicControlInputSequence[T: int, D_u: int]:
    array: Float[JaxArray, "T D_u"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return np.asarray(self.array, dtype=dtype)

    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L
    ) -> "BasicControlInputSequence[L, D_u]":
        return BasicControlInputSequence[L, D_u](array=array)

    @property
    def dimension(self) -> D_u:
        return cast(D_u, self.array.shape[1])


@jaxtyped
@dataclass(frozen=True)
class BasicControlInputBatch[T: int, D_u: int, M: int]:
    array: Float[JaxArray, "T D_u M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        return np.asarray(self.array, dtype=dtype)

    @property
    def time_horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])


@jaxtyped
@dataclass(frozen=True)
class BasicCosts[T: int, M: int]:
    array: Float[JaxArray, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return np.asarray(self.array, dtype=dtype)
