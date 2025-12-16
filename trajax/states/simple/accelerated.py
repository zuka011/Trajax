from typing import cast, Self, overload
from dataclasses import dataclass

from trajax.type import DataType, jaxtyped
from trajax.mppi.accelerated import (
    State as AnyState,
    StateBatch as AnyStateBatch,
    ControlInputSequence as AnyControlInputSequence,
    ControlInputBatch as AnyControlInputBatch,
    Costs as AnyCosts,
)


from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dims

import numpy as np
import jax.numpy as jnp


@jaxtyped
@dataclass(frozen=True)
class State[D_x: int](AnyState[D_x]):
    _array: Float[JaxArray, "D_x"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return np.asarray(self.array, dtype=dtype)

    @property
    def array(self) -> Float[JaxArray, "D_x"]:
        return self._array

    @property
    def dimension(self) -> D_x:
        return cast(D_x, self.array.shape[0])


@jaxtyped
@dataclass(frozen=True)
class StateBatch[T: int, D_x: int, M: int](AnyStateBatch[T, D_x, M]):
    _array: Float[JaxArray, "T D_x M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        return np.asarray(self.array, dtype=dtype)

    @property
    def array(self) -> Float[JaxArray, "T D_x M"]:
        return self._array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_x:
        return cast(D_x, self.array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])


@jaxtyped
@dataclass(frozen=True)
class ControlInputSequence[T: int, D_u: int](AnyControlInputSequence[T, D_u]):
    _array: Float[JaxArray, "T D_u"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return np.asarray(self.array, dtype=dtype)

    @overload
    def similar(self, *, array: Float[JaxArray, "T D_u"]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L
    ) -> "ControlInputSequence[L, D_u]": ...

    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L | None = None
    ) -> "Self | ControlInputSequence[L, D_u]":
        length = length if length is not None else cast(L, array.shape[0])

        assert array.shape[0] == length, (
            f"Expected array with time horizon {length}, but got {array.shape[0]}"
        )

        return self.__class__(array)

    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        return self._array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_u:
        return cast(D_u, self.array.shape[1])


@jaxtyped
@dataclass(frozen=True)
class ControlInputBatch[T: int, D_u: int, M: int](AnyControlInputBatch[T, D_u, M]):
    _array: Float[JaxArray, "T D_u M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        return np.asarray(self.array, dtype=dtype)

    @property
    def array(self) -> Float[JaxArray, "T D_u M"]:
        return self._array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_u:
        return cast(D_u, self.array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])


@jaxtyped
@dataclass(frozen=True)
class Costs[T: int, M: int](AnyCosts[T, M]):
    _array: Float[JaxArray, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return np.asarray(self.array, dtype=dtype)

    def zero(self) -> Self:
        return self.__class__(jnp.zeros_like(self.array))

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        return self._array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[1])
