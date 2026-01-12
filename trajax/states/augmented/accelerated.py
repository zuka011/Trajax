from typing import Self, Sequence, overload, cast
from dataclasses import dataclass

from trajax.types import (
    DataType,
    JaxState,
    JaxStateSequence,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
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

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dim1, Dim2, Dim3

import jax.numpy as jnp


@dataclass(frozen=True)
class JaxAugmentedState[P: JaxState, V: JaxState](
    AugmentedState[P, V], HasPhysical[P], HasVirtual[V], JaxState
):
    inner: BaseAugmentedState[P, V]

    @staticmethod
    def of[P_: JaxState, V_: JaxState](
        *, physical: P_, virtual: V_
    ) -> "JaxAugmentedState[P_, V_]":
        return JaxAugmentedState(
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

    @property
    def array(self) -> Float[JaxArray, "D_x"]:
        return jnp.concatenate(
            [self.inner.physical.array, self.inner.virtual.array], axis=0
        )


@dataclass(frozen=True)
class JaxAugmentedStateSequence[P: JaxStateSequence, V: JaxStateSequence](
    AugmentedStateSequence[P, V], HasPhysical[P], HasVirtual[V], JaxStateSequence
):
    inner: BaseAugmentedStateSequence[P, V]

    @staticmethod
    def of[P_: JaxStateSequence, V_: JaxStateSequence](
        *, physical: P_, virtual: V_
    ) -> "JaxAugmentedStateSequence[P_, V_]":
        return JaxAugmentedStateSequence(
            BaseAugmentedStateSequence.of(physical=physical, virtual=virtual)
        )

    @staticmethod
    def of_states[
        PS: JaxState,
        PSS: JaxStateSequence,
        VS: JaxState,
        VSS: JaxStateSequence,
    ](
        *,
        physical: StateSequenceCreator[PS, PSS],
        virtual: StateSequenceCreator[VS, VSS],
    ) -> StateSequenceCreator[
        JaxAugmentedState[PS, VS], "JaxAugmentedStateSequence[PSS, VSS]"
    ]:
        """Returns a state sequence creator for augmented states."""

        def creator(
            states: Sequence[JaxAugmentedState[PS, VS]],
        ) -> "JaxAugmentedStateSequence[PSS, VSS]":
            return JaxAugmentedStateSequence.of(
                physical=physical([s.physical for s in states]),
                virtual=virtual([s.virtual for s in states]),
            )

        return creator

    def __array__(self, dtype: DataType | None = None) -> Array[Dim2]:
        return self.inner.__array__(dtype=dtype)

    def batched(self) -> "JaxAugmentedStateBatch":
        return JaxAugmentedStateBatch(self.inner.batched())

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
    def array(self) -> Float[JaxArray, "T D_x"]:
        return jnp.concatenate(
            [self.inner.physical.array, self.inner.virtual.array], axis=1
        )


@dataclass(frozen=True)
class JaxAugmentedStateBatch[P: JaxStateBatch, V: JaxStateBatch](
    AugmentedStateBatch[P, V], HasPhysical[P], HasVirtual[V], JaxStateBatch
):
    inner: BaseAugmentedStateBatch[P, V]

    @staticmethod
    def of[P_: JaxStateBatch, V_: JaxStateBatch](
        *, physical: P_, virtual: V_
    ) -> "JaxAugmentedStateBatch[P_, V_]":
        return JaxAugmentedStateBatch(
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
    def array(self) -> Float[JaxArray, "T D_x M"]:
        return jnp.concatenate(
            [self.inner.physical.array, self.inner.virtual.array], axis=1
        )


@dataclass(frozen=True)
class JaxAugmentedControlInputSequence[
    P: JaxControlInputSequence,
    V: JaxControlInputSequence,
](
    AugmentedControlInputSequence[P, V],
    HasPhysical[P],
    HasVirtual[V],
    JaxControlInputSequence,
):
    inner: BaseAugmentedControlInputSequence[P, V]

    @staticmethod
    def of[P_: JaxControlInputSequence, V_: JaxControlInputSequence](
        *, physical: P_, virtual: V_
    ) -> "JaxAugmentedControlInputSequence[P_, V_]":
        return JaxAugmentedControlInputSequence(
            BaseAugmentedControlInputSequence.of(physical=physical, virtual=virtual)
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dim2]:
        return self.inner.__array__(dtype=dtype)

    @overload
    def similar(self, *, array: Float[JaxArray, "T D_u"]) -> Self: ...

    @overload
    def similar(
        self, *, array: Float[JaxArray, "T D_u"], length: int
    ) -> "JaxAugmentedControlInputSequence[P, V]": ...

    def similar(
        self, *, array: Float[JaxArray, "T D_u"], length: int | None = None
    ) -> "Self | JaxAugmentedControlInputSequence[P, V]":
        assert length is None or length == array.shape[0], (
            f"Length mismatch in {self.__class__.__name__}.similar: "
            f"got {array.shape[0]} but expected {length}"
        )

        physical_dim = self.inner.physical.dimension
        virtual_dim = self.inner.virtual.dimension

        return self.__class__(
            cast(
                BaseAugmentedControlInputSequence[P, V],
                BaseAugmentedControlInputSequence.of(
                    physical=self.inner.physical.similar(
                        array=jnp.asarray(array[:, :physical_dim])
                    ),
                    virtual=self.inner.virtual.similar(
                        array=jnp.asarray(array[:, -virtual_dim:])
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
    def array(self) -> Float[JaxArray, "T D_u"]:
        return jnp.concatenate(
            [self.inner.physical.array, self.inner.virtual.array], axis=1
        )


@dataclass(frozen=True)
class JaxAugmentedControlInputBatch[
    P: JaxControlInputBatch,
    V: JaxControlInputBatch,
](
    AugmentedControlInputBatch[P, V],
    HasPhysical[P],
    HasVirtual[V],
    JaxControlInputBatch,
):
    inner: BaseAugmentedControlInputBatch[P, V]

    @staticmethod
    def of[P_: JaxControlInputBatch, V_: JaxControlInputBatch](
        *, physical: P_, virtual: V_
    ) -> "JaxAugmentedControlInputBatch[P_, V_]":
        return JaxAugmentedControlInputBatch(
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
    def array(self) -> Float[JaxArray, "T D_u M"]:
        return jnp.concatenate(
            [self.inner.physical.array, self.inner.virtual.array], axis=1
        )
