from typing import Self, overload, cast
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

from numtypes import Array, Dims


@dataclass(frozen=True)
class NumPyAugmentedState[P: NumPyState, V: NumPyState, D_x: int](
    AugmentedState[P, V, D_x], NumPyState[D_x]
):
    inner: AugmentedState[P, V, D_x]

    @staticmethod
    def of[P_: NumPyState, V_: NumPyState, D_x_: int = int](
        *,
        physical: P_,
        virtual: V_,
        dimension: D_x_ | None = None,
    ) -> "NumPyAugmentedState[P_, V_, D_x_]":
        return NumPyAugmentedState(
            BaseAugmentedState.of(
                physical=physical, virtual=virtual, dimension=dimension
            )
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return self.inner.__array__(dtype=dtype)

    @property
    def physical(self) -> P:
        return self.inner.physical

    @property
    def virtual(self) -> V:
        return self.inner.virtual

    @property
    def dimension(self) -> D_x:
        return self.inner.dimension


@dataclass(frozen=True)
class NumPyAugmentedStateBatch[
    P: NumPyStateBatch,
    V: NumPyStateBatch,
    T: int,
    D_x: int,
    M: int,
](AugmentedStateBatch[P, V, T, D_x, M], NumPyStateBatch[T, D_x, M]):
    inner: AugmentedStateBatch[P, V, T, D_x, M]

    @staticmethod
    def of[
        P_: NumPyStateBatch,
        V_: NumPyStateBatch,
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
    ) -> "NumPyAugmentedStateBatch[P_, V_, T_, D_x_, M_]":
        return NumPyAugmentedStateBatch(
            BaseAugmentedStateBatch.of(
                physical=physical,
                virtual=virtual,
                horizon=horizon,
                dimension=dimension,
                rollout_count=rollout_count,
            )
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        return self.inner.__array__(dtype=dtype)

    @property
    def physical(self) -> P:
        return self.inner.physical

    @property
    def virtual(self) -> V:
        return self.inner.virtual

    @property
    def horizon(self) -> T:
        return self.inner.horizon

    @property
    def dimension(self) -> D_x:
        return self.inner.dimension

    @property
    def rollout_count(self) -> M:
        return self.inner.rollout_count


@dataclass(frozen=True)
class NumPyAugmentedControlInputSequence[
    P: NumPyControlInputSequence,
    V: NumPyControlInputSequence,
    T: int,
    D_u: int,
](AugmentedControlInputSequence[P, V, T, D_u], NumPyControlInputSequence[T, D_u]):
    inner: AugmentedControlInputSequence[P, V, T, D_u]

    @staticmethod
    def of[
        P_: NumPyControlInputSequence,
        V_: NumPyControlInputSequence,
        T_: int = int,
        D_u_: int = int,
    ](
        *,
        physical: P_,
        virtual: V_,
        horizon: T_ | None = None,
        dimension: D_u_ | None = None,
    ) -> "NumPyAugmentedControlInputSequence[P_, V_, T_, D_u_]":
        return NumPyAugmentedControlInputSequence(
            BaseAugmentedControlInputSequence.of(
                physical=physical, virtual=virtual, horizon=horizon, dimension=dimension
            )
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return self.inner.__array__(dtype=dtype)

    @overload
    def similar(self, *, array: Array[Dims[T, D_u]]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]], length: L
    ) -> "NumPyAugmentedControlInputSequence[P, V, L, D_u]": ...

    def similar[L: int](
        self, *, array: Array[Dims[T, D_u]], length: L | None = None
    ) -> "Self | NumPyAugmentedControlInputSequence[P, V, L, D_u]":
        return self.__class__(
            BaseAugmentedControlInputSequence.of(
                physical=self.inner.physical.similar(
                    array=array[:, : self.inner.physical.dimension]
                ),
                virtual=self.inner.virtual.similar(
                    array=array[:, -self.inner.virtual.dimension :]
                ),
                # NOTE: "Wrong" cast to silence the type checker.
                horizon=cast(T, length if length is not None else self.horizon),
                dimension=self.inner.dimension,
            )
        )

    @property
    def physical(self) -> P:
        return self.inner.physical

    @property
    def virtual(self) -> V:
        return self.inner.virtual

    @property
    def horizon(self) -> T:
        return self.inner.horizon

    @property
    def dimension(self) -> D_u:
        return self.inner.dimension


@dataclass(frozen=True)
class NumPyAugmentedControlInputBatch[
    P: NumPyControlInputBatch,
    V: NumPyControlInputBatch,
    T: int,
    D_u: int,
    M: int,
](AugmentedControlInputBatch[P, V, T, D_u, M], NumPyControlInputBatch[T, D_u, M]):
    inner: AugmentedControlInputBatch[P, V, T, D_u, M]

    @staticmethod
    def of[
        P_: NumPyControlInputBatch,
        V_: NumPyControlInputBatch,
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
    ) -> "NumPyAugmentedControlInputBatch[P_, V_, T_, D_u_, M_]":
        return NumPyAugmentedControlInputBatch(
            BaseAugmentedControlInputBatch.of(
                physical=physical,
                virtual=virtual,
                horizon=horizon,
                dimension=dimension,
                rollout_count=rollout_count,
            )
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        return self.inner.__array__(dtype=dtype)

    @property
    def physical(self) -> P:
        return self.inner.physical

    @property
    def virtual(self) -> V:
        return self.inner.virtual

    @property
    def horizon(self) -> T:
        return self.inner.horizon

    @property
    def dimension(self) -> D_u:
        return self.inner.dimension

    @property
    def rollout_count(self) -> M:
        return self.inner.rollout_count
