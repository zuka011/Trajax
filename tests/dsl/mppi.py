from dataclasses import dataclass
from typing import cast

from trajax.type import DataType, jaxtyped

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dims
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SimpleNumPyState[D_x: int]:
    array: Array[Dims[D_x]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return self.array


@dataclass(frozen=True)
class SimpleNumPyStateBatch[T: int, D_x: int, M: int]:
    array: Array[Dims[T, D_x, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        return self.array


@dataclass(frozen=True)
class SimpleNumPyControlInputSequence[T: int, D_u: int]:
    array: Array[Dims[T, D_u]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return self.array

    def similar(
        self, *, array: Array[Dims[T, D_u]]
    ) -> "SimpleNumPyControlInputSequence[T, D_u]":
        return SimpleNumPyControlInputSequence(array=array)


@dataclass(frozen=True)
class SimpleNumPyControlInputBatch[T: int, D_u: int, M: int]:
    array: Array[Dims[T, D_u, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        return self.array

    @property
    def time_horizon(self) -> T:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]


@jaxtyped
@dataclass(frozen=True)
class SimpleJaxState[D_x: int]:
    array: Float[JaxArray, "D_x"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return np.asarray(self.array, dtype=dtype)


@jaxtyped
@dataclass(frozen=True)
class SimpleJaxStateBatch[T: int, D_x: int, M: int]:
    array: Float[JaxArray, "T D_x M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        return np.asarray(self.array, dtype=dtype)


@jaxtyped
@dataclass(frozen=True)
class SimpleJaxControlInputSequence[T: int, D_u: int]:
    array: Float[JaxArray, "T D_u"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return np.asarray(self.array, dtype=dtype)

    def similar(
        self, *, array: Float[JaxArray, "T D_u"]
    ) -> "SimpleJaxControlInputSequence[T, D_u]":
        return SimpleJaxControlInputSequence[T, D_u](array=array)


@jaxtyped
@dataclass(frozen=True)
class SimpleJaxControlInputBatch[T: int, D_u: int, M: int]:
    array: Float[JaxArray, "T D_u M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        return np.asarray(self.array, dtype=dtype)

    @property
    def time_horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])


class numpy:
    @staticmethod
    def state[D_x: int](array: Array[Dims[D_x]]) -> SimpleNumPyState[D_x]:
        return SimpleNumPyState(array)

    @staticmethod
    def state_batch[T: int, D_x: int, M: int](
        array: Array[Dims[T, D_x, M]],
    ) -> SimpleNumPyStateBatch[T, D_x, M]:
        return SimpleNumPyStateBatch(array)

    @staticmethod
    def control_input_sequence[T: int, D_u: int](
        array: Array[Dims[T, D_u]],
    ) -> SimpleNumPyControlInputSequence[T, D_u]:
        return SimpleNumPyControlInputSequence(array)

    @staticmethod
    def control_input_batch[T: int, D_u: int, M: int](
        array: Array[Dims[T, D_u, M]],
    ) -> SimpleNumPyControlInputBatch[T, D_u, M]:
        return SimpleNumPyControlInputBatch(array)


class jax:
    @staticmethod
    def state[D_x: int](array: Array[Dims[D_x]]) -> SimpleJaxState[D_x]:
        return SimpleJaxState(jnp.asarray(array))

    @staticmethod
    def state_batch[T: int, D_x: int, M: int](
        array: Array[Dims[T, D_x, M]],
    ) -> SimpleJaxStateBatch[T, D_x, M]:
        return SimpleJaxStateBatch(jnp.asarray(array))

    @staticmethod
    def control_input_sequence[T: int, D_u: int](
        array: Array[Dims[T, D_u]],
    ) -> SimpleJaxControlInputSequence[T, D_u]:
        return SimpleJaxControlInputSequence(jnp.asarray(array))

    @staticmethod
    def control_input_batch[T: int, D_u: int, M: int](
        array: Array[Dims[T, D_u, M]] | Float[JaxArray, "T D_u M"],
    ) -> SimpleJaxControlInputBatch[T, D_u, M]:
        return SimpleJaxControlInputBatch(jnp.asarray(array))
