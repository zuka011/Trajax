from dataclasses import dataclass

from trajax.type import DataType

from numtypes import Array, Dims


@dataclass(frozen=True)
class BasicState[D_x: int]:
    array: Array[Dims[D_x]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return self.array

    @property
    def dimension(self) -> D_x:
        return self.array.shape[0]


@dataclass(frozen=True)
class BasicStateBatch[T: int, D_x: int, M: int]:
    array: Array[Dims[T, D_x, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]


@dataclass(frozen=True)
class BasicControlInputSequence[T: int, D_u: int]:
    array: Array[Dims[T, D_u]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return self.array

    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]]
    ) -> "BasicControlInputSequence[L, D_u]":
        return BasicControlInputSequence(array=array)

    @property
    def dimension(self) -> D_u:
        return self.array.shape[1]


@dataclass(frozen=True)
class BasicControlInputBatch[T: int, D_u: int, M: int]:
    array: Array[Dims[T, D_u, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        return self.array

    @property
    def time_horizon(self) -> T:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]


@dataclass(frozen=True)
class BasicCosts[T: int, M: int]:
    array: Array[Dims[T, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return self.array
