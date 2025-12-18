from dataclasses import dataclass

from trajax.type import DataType
from trajax.mppi import State, StateBatch, ControlInputSequence, ControlInputBatch
from trajax.states.augmented.common import (
    AugmentedState,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
)

from numtypes import Array, Dim1, Dim2, Dim3

import numpy as np


@dataclass(kw_only=True, frozen=True)
class BaseAugmentedState[P: State, V: State](AugmentedState[P, V]):
    _physical: P
    _virtual: V

    @staticmethod
    def of[P_: State, V_: State](
        *, physical: P_, virtual: V_
    ) -> "AugmentedState[P_, V_]":
        return BaseAugmentedState(_physical=physical, _virtual=virtual)

    def __array__(self, dtype: DataType | None = None) -> Array[Dim1]:
        return np.concatenate(
            [
                np.asarray(self.physical, dtype=dtype),
                np.asarray(self.virtual, dtype=dtype),
            ]
        )

    @property
    def physical(self) -> P:
        return self._physical

    @property
    def virtual(self) -> V:
        return self._virtual

    @property
    def dimension(self) -> int:
        return self.physical.dimension + self.virtual.dimension


@dataclass(kw_only=True, frozen=True)
class BaseAugmentedStateBatch[P: StateBatch, V: StateBatch](AugmentedStateBatch[P, V]):
    _physical: P
    _virtual: V

    @staticmethod
    def of[P_: StateBatch, V_: StateBatch](
        *, physical: P_, virtual: V_
    ) -> "AugmentedStateBatch[P_, V_]":
        return BaseAugmentedStateBatch(
            _physical=physical,
            _virtual=virtual,
        )

    def __post_init__(self) -> None:
        assert self.physical.horizon == self.virtual.horizon, (
            f"Horizon mismatch in {self.__class__.__name__}: "
            f"Got {self.physical.horizon} (physical) and {self.virtual.horizon} (virtual)"
        )
        assert self.physical.rollout_count == self.virtual.rollout_count, (
            f"Rollout count mismatch in {self.__class__.__name__}: "
            f"Got {self.physical.rollout_count} (physical) and {self.virtual.rollout_count} (virtual)"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dim3]:
        return np.concatenate(
            [
                np.asarray(self.physical, dtype=dtype),
                np.asarray(self.virtual, dtype=dtype),
            ],
            axis=1,
        )

    @property
    def physical(self) -> P:
        return self._physical

    @property
    def virtual(self) -> V:
        return self._virtual

    @property
    def horizon(self) -> int:
        return self.physical.horizon

    @property
    def dimension(self) -> int:
        return self.physical.dimension + self.virtual.dimension

    @property
    def rollout_count(self) -> int:
        return self.physical.rollout_count


@dataclass(kw_only=True, frozen=True)
class BaseAugmentedControlInputSequence[
    P: ControlInputSequence,
    V: ControlInputSequence,
](AugmentedControlInputSequence[P, V]):
    _physical: P
    _virtual: V

    @staticmethod
    def of[P_: ControlInputSequence, V_: ControlInputSequence](
        *, physical: P_, virtual: V_
    ) -> "AugmentedControlInputSequence[P_, V_]":
        return BaseAugmentedControlInputSequence(_physical=physical, _virtual=virtual)

    def __post_init__(self) -> None:
        assert self.physical.horizon == self.virtual.horizon, (
            f"Horizon mismatch in {self.__class__.__name__}: "
            f"Got {self.physical.horizon} (physical) and {self.virtual.horizon} (virtual)"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dim2]:
        return np.concatenate(
            [
                np.asarray(self.physical, dtype=dtype),
                np.asarray(self.virtual, dtype=dtype),
            ],
            axis=1,
        )

    @property
    def physical(self) -> P:
        return self._physical

    @property
    def virtual(self) -> V:
        return self._virtual

    @property
    def horizon(self) -> int:
        return self.physical.horizon

    @property
    def dimension(self) -> int:
        return self.physical.dimension + self.virtual.dimension


@dataclass(kw_only=True, frozen=True)
class BaseAugmentedControlInputBatch[P: ControlInputBatch, V: ControlInputBatch](
    AugmentedControlInputBatch[P, V]
):
    _physical: P
    _virtual: V

    @staticmethod
    def of[P_: ControlInputBatch, V_: ControlInputBatch](
        *, physical: P_, virtual: V_
    ) -> "AugmentedControlInputBatch[P_, V_]":
        return BaseAugmentedControlInputBatch(_physical=physical, _virtual=virtual)

    def __post_init__(self) -> None:
        assert self.physical.horizon == self.virtual.horizon, (
            f"Horizon mismatch in {self.__class__.__name__}: "
            f"Got {self.physical.horizon} (physical) and {self.virtual.horizon} (virtual)"
        )
        assert self.physical.rollout_count == self.virtual.rollout_count, (
            f"Rollout count mismatch in {self.__class__.__name__}: "
            f"Got {self.physical.rollout_count} (physical) and {self.virtual.rollout_count} (virtual)"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dim3]:
        return np.concatenate(
            [
                np.asarray(self.physical, dtype=dtype),
                np.asarray(self.virtual, dtype=dtype),
            ],
            axis=1,
        )

    @property
    def physical(self) -> P:
        return self._physical

    @property
    def virtual(self) -> V:
        return self._virtual

    @property
    def horizon(self) -> int:
        return self.physical.horizon

    @property
    def dimension(self) -> int:
        return self.physical.dimension + self.virtual.dimension

    @property
    def rollout_count(self) -> int:
        return self.physical.rollout_count
