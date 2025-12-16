from typing import Self, overload, cast
from dataclasses import dataclass

from trajax.type import DataType
from trajax.mppi.basic import (
    ControlInputSequence,
)
from trajax.states.augmented.common import (
    AugmentedControlInputSequence as AnyAugmentedControlInputSequence,
)
from trajax.states.augmented.base import BaseAugmentedControlInputSequence

from numtypes import Array, Dims


@dataclass(frozen=True)
class AugmentedControlInputSequence[
    PhysicalT: ControlInputSequence,
    VirtualT: ControlInputSequence,
    T: int = int,
    D_u: int = int,
](ControlInputSequence[T, D_u]):
    inner: AnyAugmentedControlInputSequence[PhysicalT, VirtualT, T, D_u]

    @staticmethod
    def of[
        P: ControlInputSequence,
        V: ControlInputSequence,
        T_: int = int,
        D_u_: int = int,
    ](
        *,
        physical: P,
        virtual: V,
        horizon: T_ | None = None,
        dimension: D_u_ | None = None,
    ) -> "AugmentedControlInputSequence[P, V, T_, D_u_]":
        return AugmentedControlInputSequence(
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
    ) -> "AugmentedControlInputSequence[PhysicalT, VirtualT, L, D_u]": ...

    def similar[L: int](
        self, *, array: Array[Dims[T, D_u]], length: L | None = None
    ) -> "Self | AugmentedControlInputSequence[PhysicalT, VirtualT, L, D_u]":
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
    def physical(self) -> PhysicalT:
        return self.inner.physical

    @property
    def virtual(self) -> VirtualT:
        return self.inner.virtual

    @property
    def horizon(self) -> T:
        return self.inner.horizon

    @property
    def dimension(self) -> D_u:
        return self.inner.dimension


# @dataclass(kw_only=True, frozen=True)
# class AugmentedSampler[
#     PhysicalSequenceT: ControlInputSequence,
#     PhysicalBatchT: ControlInputBatch,
#     VirtualSequenceT: ControlInputSequence,
#     VirtualBatchT: ControlInputBatch,
#     T: int = int,
#     D_u: int = int,
#     M: int = int,
# ](
#     Sampler[
#         AugmentedControlInputSequence[PhysicalSequenceT, VirtualSequenceT, T, D_u],
#         AugmentedControlInputBatch[PhysicalBatchT, VirtualBatchT, T, D_u, M],
#     ]
# ):
#     physical: Sampler[PhysicalSequenceT, PhysicalBatchT]
#     virtual: Sampler[VirtualSequenceT, VirtualBatchT]

#     _rollout_count: M

#     @staticmethod
#     def of[
#         PS: ControlInputSequence,
#         PB: ControlInputBatch,
#         VS: ControlInputSequence,
#         VB: ControlInputBatch,
#         T_: int = int,
#         D_u_: int = int,
#         M_: int = int,
#     ](
#         *,
#         physical: Sampler[PS, PB, M_],
#         virtual: Sampler[VS, VB, M_],
#         horizon: T_ | None = None,
#         dimension: D_u_ | None = None,
#         rollout_count: M_ | None = None,
#     ) -> "AugmentedSampler[PS, PB, VS, VB, T_, D_u_, M_]":
#         return AugmentedSampler(
#             physical=physical,
#             virtual=virtual,
#             _rollout_count=cast(
#                 M_,
#                 rollout_count if rollout_count is not None else physical.rollout_count,
#             ),
#         )

#     def __post_init__(self) -> None:
#         assert (
#             self.rollout_count
#             == self.physical.rollout_count
#             == self.virtual.rollout_count
#         ), (
#             f"Rollout count mismatch in {self.__class__.__name__}: "
#             f"expected {self.rollout_count}, got "
#             f"{self.physical.rollout_count} (physical) and {self.virtual.rollout_count} (virtual)"
#         )

#     def sample(
#         self,
#         *,
#         around: AugmentedControlInputSequence[
#             PhysicalSequenceT, VirtualSequenceT, T, D_u
#         ],
#     ) -> AugmentedControlInputBatch[PhysicalBatchT, VirtualBatchT, T, D_u, M]:
#         physical_batch = self.physical.sample(around=around.physical)
#         virtual_batch = self.virtual.sample(around=around.virtual)

#         return AugmentedControlInputBatch.of(
#             physical=physical_batch,
#             virtual=virtual_batch,
#             horizon=around.horizon,
#             dimension=around.dimension,
#             rollout_count=self.rollout_count,
#         )

#     @property
#     def rollout_count(self) -> M:
#         return self._rollout_count
