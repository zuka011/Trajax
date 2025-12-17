from typing import Self, overload, cast, Sequence
from dataclasses import dataclass

from trajax.type import DataType
from trajax.mppi import (
    NumPyState,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyCosts,
)

import numpy as np
from numtypes import Array, Dims, D, shape_of


@dataclass(frozen=True)
class NumPySimpleState[D_x: int](NumPyState[D_x]):
    array: Array[Dims[D_x]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return self.array

    @property
    def dimension(self) -> D_x:
        return self.array.shape[0]


@dataclass(frozen=True)
class NumPySimpleStateBatch[T: int, D_x: int, M: int](NumPyStateBatch[T, D_x, M]):
    array: Array[Dims[T, D_x, M]]

    @staticmethod
    def of_states[D_x_: int, T_: int = int](
        states: Sequence[NumPySimpleState[D_x_]], *, horizon: T_ | None = None
    ) -> "NumPySimpleStateBatch[T_, D_x_, D[1]]":
        """Creates a simple state batch from a sequence of simple states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        dimension = states[0].dimension

        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        assert shape_of(array, matches=(horizon, dimension, 1))

        return NumPySimpleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_x:
        return self.array.shape[1]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]


@dataclass(frozen=True)
class NumPySimpleControlInputSequence[T: int, D_u: int](
    NumPyControlInputSequence[T, D_u]
):
    array: Array[Dims[T, D_u]]

    @staticmethod
    def zeroes[T_: int, D_u_: int](
        horizon: T_, dimension: D_u_
    ) -> "NumPySimpleControlInputSequence[T_, D_u_]":
        """Creates a zeroed control input sequence for the given horizon."""
        array = np.zeros((horizon, dimension))

        assert shape_of(array, matches=(horizon, dimension))

        return NumPySimpleControlInputSequence(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return self.array

    @overload
    def similar(self, *, array: Array[Dims[T, D_u]]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]], length: L
    ) -> "NumPySimpleControlInputSequence[L, D_u]": ...

    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]], length: L | None = None
    ) -> "Self | NumPySimpleControlInputSequence[L, D_u]":
        assert length is None or length == array.shape[0], (
            f"Length mismatch: expected {length}, got {array.shape[0]}"
        )

        # NOTE: "Wrong" cast to silence the type checker.
        return self.__class__(cast(Array[Dims[T, D_u]], array))

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_u:
        return self.array.shape[1]


@dataclass(frozen=True)
class NumPySimpleControlInputBatch[T: int, D_u: int, M: int](
    NumPyControlInputBatch[T, D_u, M]
):
    array: Array[Dims[T, D_u, M]]

    @staticmethod
    def zero[T_: int, D_u_: int, M_: int](
        *, horizon: T_, dimension: D_u_, rollout_count: M_ = 1
    ) -> "NumPySimpleControlInputBatch[T_, D_u_, M_]":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        array = np.zeros((horizon, dimension, rollout_count))

        assert shape_of(array, matches=(horizon, dimension, rollout_count))

        return NumPySimpleControlInputBatch(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_u:
        return self.array.shape[1]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]


@dataclass(frozen=True)
class NumPySimpleCosts[T: int = int, M: int = int](NumPyCosts[T, M]):
    array: Array[Dims[T, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return self.array

    def similar(self, *, array: Array[Dims[T, M]]) -> Self:
        return self.__class__(array)

    def zero(self) -> Self:
        return self.__class__(np.zeros_like(self.array))

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[1]
