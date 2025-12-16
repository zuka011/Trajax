from typing import Self, overload, cast
from dataclasses import dataclass

from trajax.type import DataType
from trajax.mppi.basic import (
    State as AnyState,
    StateBatch as AnyStateBatch,
    ControlInputSequence as AnyControlInputSequence,
    ControlInputBatch as AnyControlInputBatch,
    Costs as AnyCosts,
)

import numpy as np
from numtypes import Array, Dims, shape_of


@dataclass(frozen=True)
class State[D_x: int](AnyState[D_x]):
    array: Array[Dims[D_x]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return self.array

    @property
    def dimension(self) -> D_x:
        return self.array.shape[0]


@dataclass(frozen=True)
class StateBatch[T: int, D_x: int, M: int](AnyStateBatch[T, D_x, M]):
    array: Array[Dims[T, D_x, M]]

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
class ControlInputSequence[T: int, D_u: int](AnyControlInputSequence[T, D_u]):
    array: Array[Dims[T, D_u]]

    @staticmethod
    def zeroes[T_: int, D_u_: int](
        horizon: T_, dimension: D_u_
    ) -> "ControlInputSequence[T_, D_u_]":
        """Creates a zeroed control input sequence for the given horizon."""
        array = np.zeros((horizon, dimension))

        assert shape_of(array, matches=(horizon, dimension))

        return ControlInputSequence(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return self.array

    @overload
    def similar(self, *, array: Array[Dims[T, D_u]]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]], length: L
    ) -> "ControlInputSequence[L, D_u]": ...

    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]], length: L | None = None
    ) -> "Self | ControlInputSequence[L, D_u]":
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
class ControlInputBatch[T: int, D_u: int, M: int](AnyControlInputBatch[T, D_u, M]):
    array: Array[Dims[T, D_u, M]]

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
class Costs[T: int = int, M: int = int](AnyCosts[T, M]):
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
