from typing import Self, overload, cast, Sequence
from dataclasses import dataclass

from trajax.types import (
    DataType,
    NumPyState,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyCosts,
    NumPyIntegratorObstacleStates,
    NumPyIntegratorObstacleStateSequences,
    NumPyIntegratorObstacleVelocities,
    NumPyIntegratorObstacleControlInputSequences,
)

import numpy as np
from numtypes import Array, Dims, D, shape_of


@dataclass(frozen=True)
class NumPySimpleState[D_x: int](NumPyState[D_x]):
    _array: Array[Dims[D_x]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return self.array

    @property
    def dimension(self) -> D_x:
        return self.array.shape[0]

    @property
    def array(self) -> Array[Dims[D_x]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleStateBatch[T: int, D_x: int, M: int](NumPyStateBatch[T, D_x, M]):
    _array: Array[Dims[T, D_x, M]]

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

    @property
    def array(self) -> Array[Dims[T, D_x, M]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleControlInputSequence[T: int, D_u: int](
    NumPyControlInputSequence[T, D_u]
):
    _array: Array[Dims[T, D_u]]

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

    @property
    def array(self) -> Array[Dims[T, D_u]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleControlInputBatch[T: int, D_u: int, M: int](
    NumPyControlInputBatch[T, D_u, M]
):
    _array: Array[Dims[T, D_u, M]]

    @staticmethod
    def zero[T_: int, D_u_: int, M_: int](
        *, horizon: T_, dimension: D_u_, rollout_count: M_ = 1
    ) -> "NumPySimpleControlInputBatch[T_, D_u_, M_]":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        array = np.zeros((horizon, dimension, rollout_count))

        assert shape_of(array, matches=(horizon, dimension, rollout_count))

        return NumPySimpleControlInputBatch(array)

    @staticmethod
    def create[T_: int, D_u_: int, M_: int](
        *, array: Array[Dims[T_, D_u_, M_]]
    ) -> "NumPySimpleControlInputBatch[T_, D_u_, M_]":
        """Creates a NumPy simple control input batch from the given array."""
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

    @property
    def array(self) -> Array[Dims[T, D_u, M]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleCosts[T: int, M: int](NumPyCosts[T, M]):
    _array: Array[Dims[T, M]]

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

    @property
    def array(self) -> Array[Dims[T, M]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleObstacleStates[D_x: int, K: int](
    NumPyIntegratorObstacleStates[D_x, K]
):
    _array: Array[Dims[D_x, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x, K]]:
        return self.array

    @property
    def dimension(self) -> D_x:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[D_x, K]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleObstacleStateSequences[T: int, D_x: int, K: int](
    NumPyIntegratorObstacleStateSequences[T, D_x, K]
):
    _array: Array[Dims[T, D_x, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_x:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, D_x, K]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleObstacleVelocities[D_v: int, K: int](
    NumPyIntegratorObstacleVelocities[D_v, K]
):
    _array: Array[Dims[D_v, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_v, K]]:
        return self.array

    @property
    def dimension(self) -> D_v:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[D_v, K]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleObstacleControlInputSequences[T: int, D_u: int, K: int](
    NumPyIntegratorObstacleControlInputSequences[T, D_u, K]
):
    _array: Array[Dims[T, D_u, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, K]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_u:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, D_u, K]]:
        return self._array
