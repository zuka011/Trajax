from typing import cast
from dataclasses import dataclass

from trajax.type import DataType
from trajax.mppi import State, StateBatch, ControlInputSequence, ControlInputBatch
from trajax.states.augmented.common import (
    AugmentedState,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
)

from numtypes import Array, Dims

import numpy as np


@dataclass(kw_only=True, frozen=True)
class BaseAugmentedState[P: State, V: State, D_x: int](AugmentedState[P, V, D_x]):
    _physical: P
    _virtual: V
    _dimension: D_x

    @staticmethod
    def of[P_: State, V_: State, D_x_: int](
        *, physical: P_, virtual: V_, dimension: D_x_ | None = None
    ) -> "AugmentedState[P_, V_, D_x_]":
        return BaseAugmentedState(
            _physical=physical,
            _virtual=virtual,
            _dimension=cast(
                D_x_,
                dimension
                if dimension is not None
                else physical.dimension + virtual.dimension,
            ),
        )

    def __post_init__(self) -> None:
        assert self.dimension == (self.physical.dimension + self.virtual.dimension), (
            f"State dimension mismatch in {self.__class__.__name__}: "
            f"expected {self.dimension}, got "
            f"{self.physical.dimension} (physical) + {self.virtual.dimension} (virtual) "
            f"= {self.physical.dimension + self.virtual.dimension}"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
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
    def dimension(self) -> D_x:
        return self._dimension


@dataclass(kw_only=True, frozen=True)
class BaseAugmentedStateBatch[
    P: StateBatch,
    V: StateBatch,
    T: int,
    D_x: int,
    M: int,
](AugmentedStateBatch[P, V, T, D_x, M]):
    _physical: P
    _virtual: V
    _horizon: T
    _dimension: D_x
    _rollout_count: M

    @staticmethod
    def of[
        P_: StateBatch,
        V_: StateBatch,
        T_: int = int,
        D_x_: int = int,
        M_: int = int,
    ](
        *,
        physical: P_,
        virtual: V_,
        horizon: T_ | None = None,
        dimension: D_x_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "AugmentedStateBatch[P_, V_, T_, D_x_, M_]":
        return BaseAugmentedStateBatch(
            _physical=physical,
            _virtual=virtual,
            _horizon=cast(
                T_,
                horizon if horizon is not None else physical.horizon,
            ),
            _dimension=cast(
                D_x_,
                dimension
                if dimension is not None
                else physical.dimension + virtual.dimension,
            ),
            _rollout_count=cast(
                M_,
                rollout_count if rollout_count is not None else physical.rollout_count,
            ),
        )

    def __post_init__(self) -> None:
        assert self.horizon == self.physical.horizon == self.virtual.horizon, (
            f"Horizon mismatch in {self.__class__.__name__}: "
            f"expected {self.horizon}, got "
            f"{self.physical.horizon} (physical) and {self.virtual.horizon} (virtual)"
        )
        assert self.dimension == (self.physical.dimension + self.virtual.dimension), (
            f"State dimension mismatch in {self.__class__.__name__}: "
            f"expected {self.dimension}, got "
            f"{self.physical.dimension} (physical) + {self.virtual.dimension} (virtual) "
            f"= {self.physical.dimension + self.virtual.dimension}"
        )
        assert (
            self.rollout_count
            == self.physical.rollout_count
            == self.virtual.rollout_count
        ), (
            f"Rollout count mismatch in {self.__class__.__name__}: "
            f"expected {self.rollout_count}, got "
            f"{self.physical.rollout_count} (physical) and {self.virtual.rollout_count} (virtual)"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
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
    def horizon(self) -> T:
        return self._horizon

    @property
    def dimension(self) -> D_x:
        return self._dimension

    @property
    def rollout_count(self) -> M:
        return self._rollout_count


@dataclass(kw_only=True, frozen=True)
class BaseAugmentedControlInputSequence[
    P: ControlInputSequence,
    V: ControlInputSequence,
    T: int,
    D_u: int,
](AugmentedControlInputSequence[P, V, T, D_u]):
    _physical: P
    _virtual: V
    _horizon: T
    _dimension: D_u

    @staticmethod
    def of[
        P_: ControlInputSequence,
        V_: ControlInputSequence,
        T_: int = int,
        D_u_: int = int,
    ](
        *,
        physical: P_,
        virtual: V_,
        horizon: T_ | None = None,
        dimension: D_u_ | None = None,
    ) -> "AugmentedControlInputSequence[P_, V_, T_, D_u_]":
        return BaseAugmentedControlInputSequence(
            _physical=physical,
            _virtual=virtual,
            _horizon=cast(
                T_,
                horizon if horizon is not None else physical.horizon,
            ),
            _dimension=cast(
                D_u_,
                dimension
                if dimension is not None
                else physical.dimension + virtual.dimension,
            ),
        )

    def __post_init__(self) -> None:
        assert self.horizon == self.physical.horizon == self.virtual.horizon, (
            f"Horizon mismatch in {self.__class__.__name__}: "
            f"expected {self.horizon}, got "
            f"{self.physical.horizon} (physical) and {self.virtual.horizon} (virtual)"
        )
        assert self.dimension == (self.physical.dimension + self.virtual.dimension), (
            f"Control dimension mismatch in {self.__class__.__name__}: "
            f"expected {self.dimension}, got "
            f"{self.physical.dimension} (physical) + {self.virtual.dimension} (virtual) "
            f"= {self.physical.dimension + self.virtual.dimension}"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
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
    def horizon(self) -> T:
        return self._horizon

    @property
    def dimension(self) -> D_u:
        return self._dimension


@dataclass(kw_only=True, frozen=True)
class BaseAugmentedControlInputBatch[
    P: ControlInputBatch,
    V: ControlInputBatch,
    T: int,
    D_u: int,
    M: int,
](AugmentedControlInputBatch[P, V, T, D_u, M]):
    _physical: P
    _virtual: V
    _horizon: T
    _dimension: D_u
    _rollout_count: M

    @staticmethod
    def of[
        P_: ControlInputBatch,
        V_: ControlInputBatch,
        T_: int = int,
        D_u_: int = int,
        M_: int = int,
    ](
        *,
        physical: P_,
        virtual: V_,
        horizon: T_ | None = None,
        dimension: D_u_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "AugmentedControlInputBatch[P_, V_, T_, D_u_, M_]":
        return BaseAugmentedControlInputBatch(
            _physical=physical,
            _virtual=virtual,
            _horizon=cast(
                T_,
                horizon if horizon is not None else physical.horizon,
            ),
            _dimension=cast(
                D_u_,
                dimension
                if dimension is not None
                else physical.dimension + virtual.dimension,
            ),
            _rollout_count=cast(
                M_,
                rollout_count if rollout_count is not None else physical.rollout_count,
            ),
        )

    def __post_init__(self) -> None:
        assert self.horizon == self.physical.horizon == self.virtual.horizon, (
            f"Horizon mismatch in {self.__class__.__name__}: "
            f"expected {self.horizon}, got "
            f"{self.physical.horizon} (physical) and {self.virtual.horizon} (virtual)"
        )
        assert self.dimension == (self.physical.dimension + self.virtual.dimension), (
            f"Control dimension mismatch in {self.__class__.__name__}: "
            f"expected {self.dimension}, got "
            f"{self.physical.dimension} (physical) + {self.virtual.dimension} (virtual) "
            f"= {self.physical.dimension + self.virtual.dimension}"
        )
        assert (
            self.rollout_count
            == self.physical.rollout_count
            == self.virtual.rollout_count
        ), (
            f"Rollout count mismatch in {self.__class__.__name__}: "
            f"expected {self.rollout_count}, got "
            f"{self.physical.rollout_count} (physical) and {self.virtual.rollout_count} (virtual)"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
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
    def horizon(self) -> T:
        return self._horizon

    @property
    def dimension(self) -> D_u:
        return self._dimension

    @property
    def rollout_count(self) -> M:
        return self._rollout_count
