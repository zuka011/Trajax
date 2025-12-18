from typing import cast, Self, overload, Sequence
from dataclasses import dataclass

from trajax.type import DataType, jaxtyped
from trajax.mppi import (
    JaxState,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxCosts,
)

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dims

import numpy as np
import jax.numpy as jnp


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleState[D_x: int](JaxState[D_x]):
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
class JaxSimpleStateBatch[T: int, D_x: int, M: int](JaxStateBatch[T, D_x, M]):
    _array: Float[JaxArray, "T D_x M"]

    @staticmethod
    def of_states[D_x_: int, T_: int = int](
        states: Sequence[JaxSimpleState[D_x_]], *, horizon: T_ | None = None
    ) -> "JaxSimpleStateBatch[T_, D_x_, int]":
        """Creates a simple state batch from a sequence of simple states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))

        array = jnp.stack([state.array for state in states], axis=0)[:, :, jnp.newaxis]

        return JaxSimpleStateBatch(array)

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
class JaxSimpleControlInputSequence[T: int, D_u: int](JaxControlInputSequence[T, D_u]):
    _array: Float[JaxArray, "T D_u"]

    @staticmethod
    def zeroes[T_: int, D_u_: int](
        horizon: T_, dimension: D_u_
    ) -> "JaxSimpleControlInputSequence[T_, D_u_]":
        """Creates a zeroed control input sequence for the given horizon."""
        array = jnp.zeros((horizon, dimension))

        return JaxSimpleControlInputSequence(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return np.asarray(self.array, dtype=dtype)

    @overload
    def similar(self, *, array: Float[JaxArray, "T D_u"]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L
    ) -> "JaxSimpleControlInputSequence[L, D_u]": ...

    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L | None = None
    ) -> "Self | JaxSimpleControlInputSequence[L, D_u]":
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
class JaxSimpleControlInputBatch[T: int, D_u: int, M: int](
    JaxControlInputBatch[T, D_u, M]
):
    _array: Float[JaxArray, "T D_u M"]

    @staticmethod
    def create(*, array: Float[JaxArray, "T D_u M"]) -> "JaxSimpleControlInputBatch":
        """Factory method to create a JAX simple control input batch from an array."""
        return JaxSimpleControlInputBatch(array)

    @staticmethod
    def zero[T_: int, D_u_: int, M_: int](
        *, horizon: T_, dimension: D_u_, rollout_count: M_ = 1
    ) -> "JaxSimpleControlInputBatch[T_, D_u_, M_]":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        return JaxSimpleControlInputBatch(
            jnp.zeros((horizon, dimension, rollout_count))
        )

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
class JaxSimpleCosts[T: int, M: int](JaxCosts[T, M]):
    _array: Float[JaxArray, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return np.asarray(self.array, dtype=dtype)

    def similar(self, *, array: Float[JaxArray, "T M"]) -> Self:
        return self.__class__(array)

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
