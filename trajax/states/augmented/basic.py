from typing import Self, Sequence, overload, cast
from dataclasses import dataclass

from trajax.types import (
    DataType,
    NumPyState,
    NumPyStateSequence,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    AugmentedState,
    AugmentedStateSequence,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
    HasPhysical,
    HasVirtual,
    StateSequenceCreator,
)
from trajax.states.augmented.base import (
    BaseAugmentedState,
    BaseAugmentedStateSequence,
    BaseAugmentedStateBatch,
    BaseAugmentedControlInputSequence,
    BaseAugmentedControlInputBatch,
)

from numtypes import Array, Dim1, Dim2, Dim3

import numpy as np


@dataclass(frozen=True)
class NumPyAugmentedState[P: NumPyState, V: NumPyState](
    AugmentedState[P, V], HasPhysical[P], HasVirtual[V], NumPyState
):
    inner: BaseAugmentedState[P, V]

    @staticmethod
    def of[P_: NumPyState, V_: NumPyState](
        *, physical: P_, virtual: V_
    ) -> "NumPyAugmentedState[P_, V_]":
        return NumPyAugmentedState(
            BaseAugmentedState.of(physical=physical, virtual=virtual)
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dim1]:
        return np.asarray(self.inner)

    @property
    def physical(self) -> P:
        return self.inner.physical

    @property
    def virtual(self) -> V:
        return self.inner.virtual

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    @property
    def array(self) -> Array[Dim1]:
        return np.concatenate(
            [self.inner.physical.array, self.inner.virtual.array], axis=0
        )


@dataclass(frozen=True)
class NumPyAugmentedStateSequence[P: NumPyStateSequence, V: NumPyStateSequence](
    AugmentedStateSequence[P, V],
    HasPhysical[P],
    HasVirtual[V],
    NumPyStateSequence,
):
    inner: BaseAugmentedStateSequence[P, V]

    @staticmethod
    def of[P_: NumPyStateSequence, V_: NumPyStateSequence](
        *, physical: P_, virtual: V_
    ) -> "NumPyAugmentedStateSequence[P_, V_]":
        return NumPyAugmentedStateSequence(
            BaseAugmentedStateSequence.of(physical=physical, virtual=virtual)
        )

    @staticmethod
    def of_states[
        PS: NumPyState,
        PSS: NumPyStateSequence,
        VS: NumPyState,
        VSS: NumPyStateSequence,
    ](
        *,
        physical: StateSequenceCreator[PS, PSS],
        virtual: StateSequenceCreator[VS, VSS],
    ) -> StateSequenceCreator[
        NumPyAugmentedState[PS, VS], "NumPyAugmentedStateSequence[PSS, VSS]"
    ]:
        """Returns a state sequence creator for augmented states."""

        def creator(
            states: Sequence[NumPyAugmentedState[PS, VS]],
        ) -> "NumPyAugmentedStateSequence[PSS, VSS]":
            return NumPyAugmentedStateSequence.of(
                physical=physical([s.physical for s in states]),
                virtual=virtual([s.virtual for s in states]),
            )

        return creator

    def __array__(self, dtype: DataType | None = None) -> Array[Dim2]:
        return self.inner.__array__(dtype=dtype)

    def batched(self) -> "NumPyAugmentedStateBatch ":
        return NumPyAugmentedStateBatch(self.inner.batched())

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
    def array(self) -> Array[Dim2]:
        return np.concatenate(
            [self.inner.physical.array, self.inner.virtual.array], axis=1
        )


@dataclass(frozen=True)
class NumPyAugmentedStateBatch[P: NumPyStateBatch, V: NumPyStateBatch](
    AugmentedStateBatch[P, V], HasPhysical[P], HasVirtual[V], NumPyStateBatch
):
    inner: BaseAugmentedStateBatch[P, V]

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

    @property
    def array(self) -> Array[Dim3]:
        return np.concatenate(
            [self.inner.physical.array, self.inner.virtual.array], axis=1
        )


@dataclass(frozen=True)
class NumPyAugmentedControlInputSequence[
    P: NumPyControlInputSequence,
    V: NumPyControlInputSequence,
](
    AugmentedControlInputSequence[P, V],
    HasPhysical[P],
    HasVirtual[V],
    NumPyControlInputSequence,
):
    inner: BaseAugmentedControlInputSequence[P, V]

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
            cast(
                BaseAugmentedControlInputSequence[P, V],
                BaseAugmentedControlInputSequence.of(
                    physical=self.inner.physical.similar(
                        array=array[:, : self.inner.physical.dimension]
                    ),
                    virtual=self.inner.virtual.similar(
                        array=array[:, -self.inner.virtual.dimension :]
                    ),
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

    @property
    def array(self) -> Array[Dim2]:
        return np.concatenate(
            [self.inner.physical.array, self.inner.virtual.array], axis=1
        )


@dataclass(frozen=True)
class NumPyAugmentedControlInputBatch[
    P: NumPyControlInputBatch,
    V: NumPyControlInputBatch,
](
    AugmentedControlInputBatch[P, V],
    HasPhysical[P],
    HasVirtual[V],
    NumPyControlInputBatch,
):
    inner: BaseAugmentedControlInputBatch[P, V]

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

    @property
    def array(self) -> Array[Dim3]:
        return np.concatenate(
            [self.inner.physical.array, self.inner.virtual.array], axis=1
        )
