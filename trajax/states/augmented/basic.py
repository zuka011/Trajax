from typing import Self, overload
from dataclasses import dataclass

from trajax.type import DataType
from trajax.mppi import (
    NumPyState,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
)
from trajax.states.augmented.common import (
    AugmentedState,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
)
from trajax.states.augmented.base import (
    BaseAugmentedState,
    BaseAugmentedStateBatch,
    BaseAugmentedControlInputSequence,
    BaseAugmentedControlInputBatch,
)

from numtypes import Array, Dim1, Dim2, Dim3


@dataclass(frozen=True)
class NumPyAugmentedState[P: NumPyState, V: NumPyState](
    AugmentedState[P, V], NumPyState
):
    inner: AugmentedState[P, V]

    @staticmethod
    def of[P_: NumPyState, V_: NumPyState](
        *, physical: P_, virtual: V_
    ) -> "NumPyAugmentedState[P_, V_]":
        return NumPyAugmentedState(
            BaseAugmentedState.of(physical=physical, virtual=virtual)
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dim1]:
        return self.inner.__array__(dtype=dtype)

    @property
    def physical(self) -> P:
        return self.inner.physical

    @property
    def virtual(self) -> V:
        return self.inner.virtual

    @property
    def dimension(self) -> int:
        return self.inner.dimension


@dataclass(frozen=True)
class NumPyAugmentedStateBatch[P: NumPyStateBatch, V: NumPyStateBatch](
    AugmentedStateBatch[P, V], NumPyStateBatch
):
    inner: AugmentedStateBatch[P, V]

    @staticmethod
    def of[P_: NumPyStateBatch, V_: NumPyStateBatch](
        *, physical: P_, virtual: V_
    ) -> "NumPyAugmentedStateBatch[P_, V_]":
        return NumPyAugmentedStateBatch(
            BaseAugmentedStateBatch.of(physical=physical, virtual=virtual)
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dim3]:
        return self.inner.__array__(dtype=dtype)

    @property
    def physical(self) -> P:
        return self.inner.physical

    @property
    def virtual(self) -> V:
        return self.inner.virtual

    @property
    def horizon(self) -> int:
        return self.inner.horizon

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    @property
    def rollout_count(self) -> int:
        return self.inner.rollout_count


@dataclass(frozen=True)
class NumPyAugmentedControlInputSequence[
    P: NumPyControlInputSequence,
    V: NumPyControlInputSequence,
](AugmentedControlInputSequence[P, V], NumPyControlInputSequence):
    inner: AugmentedControlInputSequence[P, V]

    @staticmethod
    def of[P_: NumPyControlInputSequence, V_: NumPyControlInputSequence](
        *, physical: P_, virtual: V_
    ) -> "NumPyAugmentedControlInputSequence[P_, V_]":
        return NumPyAugmentedControlInputSequence(
            BaseAugmentedControlInputSequence.of(physical=physical, virtual=virtual)
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dim2]:
        return self.inner.__array__(dtype=dtype)

    @overload
    def similar(self, *, array: Array[Dim2]) -> Self: ...

    @overload
    def similar(
        self, *, array: Array[Dim2], length: int
    ) -> "NumPyAugmentedControlInputSequence[P, V]": ...

    def similar(
        self, *, array: Array[Dim2], length: int | None = None
    ) -> "Self | NumPyAugmentedControlInputSequence[P, V]":
        assert length is None or length == array.shape[0], (
            f"Length mismatch in {self.__class__.__name__}.similar: "
            f"got {array.shape[0]} but expected {length}"
        )

        return self.__class__(
            BaseAugmentedControlInputSequence.of(
                physical=self.inner.physical.similar(
                    array=array[:, : self.inner.physical.dimension]
                ),
                virtual=self.inner.virtual.similar(
                    array=array[:, -self.inner.virtual.dimension :]
                ),
            )
        )

    @property
    def physical(self) -> P:
        return self.inner.physical

    @property
    def virtual(self) -> V:
        return self.inner.virtual

    @property
    def horizon(self) -> int:
        return self.inner.horizon

    @property
    def dimension(self) -> int:
        return self.inner.dimension


@dataclass(frozen=True)
class NumPyAugmentedControlInputBatch[
    P: NumPyControlInputBatch,
    V: NumPyControlInputBatch,
](AugmentedControlInputBatch[P, V], NumPyControlInputBatch):
    inner: AugmentedControlInputBatch[P, V]

    @staticmethod
    def of[P_: NumPyControlInputBatch, V_: NumPyControlInputBatch](
        *, physical: P_, virtual: V_
    ) -> "NumPyAugmentedControlInputBatch[P_, V_]":
        return NumPyAugmentedControlInputBatch(
            BaseAugmentedControlInputBatch.of(physical=physical, virtual=virtual)
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dim3]:
        return self.inner.__array__(dtype=dtype)

    @property
    def physical(self) -> P:
        return self.inner.physical

    @property
    def virtual(self) -> V:
        return self.inner.virtual

    @property
    def horizon(self) -> int:
        return self.inner.horizon

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    @property
    def rollout_count(self) -> int:
        return self.inner.rollout_count
