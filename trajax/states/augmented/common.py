import asyncio
from typing import Callable, Protocol
from dataclasses import dataclass

from trajax.mppi import (
    State,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
    DynamicalModel,
    Sampler,
)


class AugmentedState[PhysicalT: State, VirtualT: State, D_x: int](State[D_x], Protocol):
    @property
    def physical(self) -> PhysicalT:
        """Returns the physical part of the augmented state."""
        ...

    @property
    def virtual(self) -> VirtualT:
        """Returns the virtual part of the augmented state."""
        ...


class AugmentedStateBatch[P: StateBatch, V: StateBatch, T: int, D_x: int, M: int](
    StateBatch[T, D_x, M], Protocol
):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented state batch."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented state batch."""
        ...


class AugmentedControlInputSequence[
    P: ControlInputSequence,
    V: ControlInputSequence,
    T: int = int,
    D_u: int = int,
](ControlInputSequence[T, D_u], Protocol):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented control input sequence."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented control input sequence."""
        ...


class AugmentedControlInputBatch[
    P: StateBatch,
    V: StateBatch,
    T: int = int,
    D_u: int = int,
    M: int = int,
](ControlInputBatch[T, D_u, M], Protocol):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented control input batch."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented control input batch."""
        ...


class AugmentedStateCreator[P: State, V: State, A: AugmentedState, D_x: int](Protocol):
    def of(self, *, physical: P, virtual: V, dimension: D_x) -> A:
        """Creates an augmented state from physical and virtual parts."""
        ...


class AugmentedStateBatchCreator[
    P: StateBatch,
    V: StateBatch,
    A: AugmentedStateBatch,
    T: int,
    D_x: int,
    M: int,
](Protocol):
    def of(
        self, *, physical: P, virtual: V, horizon: T, dimension: D_x, rollout_count: M
    ) -> A:
        """Creates an augmented state batch from physical and virtual parts."""
        ...


class AugmentedControlInputBatchCreator[
    P: ControlInputBatch,
    V: ControlInputBatch,
    A: AugmentedControlInputBatch,
](Protocol):
    def of(
        self,
        *,
        physical: P,
        virtual: V,
        horizon: int,
        dimension: int,
        rollout_count: int,
    ) -> A:
        """Creates an augmented control input batch from physical and virtual parts."""
        ...


@dataclass(kw_only=True, frozen=True)
class AugmentedModel[
    PInStateT: State,
    POutStateT: State,
    PStateBatchT: StateBatch,
    PControlInputSequenceT: ControlInputSequence,
    PControlInputBatchT: ControlInputBatch,
    VInStateT: State,
    VOutStateT: State,
    VStateBatchT: StateBatch,
    VControlInputSequenceT: ControlInputSequence,
    VControlInputBatchT: ControlInputBatch,
    AStateT: AugmentedState,
    AStateBatchT: AugmentedStateBatch,
    T: int,
    D_x: int,
    M: int,
](
    DynamicalModel[
        AugmentedState[PInStateT, VInStateT, D_x],
        AStateT,
        AStateBatchT,
        AugmentedControlInputSequence[
            PControlInputSequenceT, VControlInputSequenceT, T, int
        ],
        AugmentedControlInputBatch[PControlInputBatchT, VControlInputBatchT, T, int, M],
    ]
):
    physical: DynamicalModel[
        PInStateT, POutStateT, PStateBatchT, PControlInputSequenceT, PControlInputBatchT
    ]
    virtual: DynamicalModel[
        VInStateT, VOutStateT, VStateBatchT, VControlInputSequenceT, VControlInputBatchT
    ]

    state: AugmentedStateCreator[POutStateT, VOutStateT, AStateT, D_x]
    batch: AugmentedStateBatchCreator[
        PStateBatchT, VStateBatchT, AStateBatchT, T, D_x, M
    ]

    @staticmethod
    def of[
        PIS: State,
        POS: State,
        PSB: StateBatch,
        PCIS: ControlInputSequence,
        PCIB: ControlInputBatch,
        VIS: State,
        VOS: State,
        VSB: StateBatch,
        VCIS: ControlInputSequence,
        VCIB: ControlInputBatch,
        AS: AugmentedState,
        ASB: AugmentedStateBatch,
        T_: int,
        D_x_: int,
        M_: int,
    ](
        *,
        physical: DynamicalModel[PIS, POS, PSB, PCIS, PCIB],
        virtual: DynamicalModel[VIS, VOS, VSB, VCIS, VCIB],
        state: AugmentedStateCreator[POS, VOS, AS, D_x_],
        batch: AugmentedStateBatchCreator[PSB, VSB, ASB, T_, D_x_, M_],
    ) -> "AugmentedModel[PIS, POS, PSB, PCIS, PCIB, VIS, VOS, VSB, VCIS, VCIB, AS, ASB, T_, D_x_, M_]":
        return AugmentedModel(
            physical=physical, virtual=virtual, state=state, batch=batch
        )

    async def simulate(
        self,
        inputs: AugmentedControlInputBatch[
            PControlInputBatchT, VControlInputBatchT, T, int, M
        ],
        initial_state: AugmentedState[PInStateT, VInStateT, D_x],
    ) -> AStateBatchT:
        physical, virtual = await asyncio.gather(
            self.physical.simulate(
                inputs=inputs.physical, initial_state=initial_state.physical
            ),
            self.virtual.simulate(
                inputs=inputs.virtual, initial_state=initial_state.virtual
            ),
        )

        return self.batch.of(
            physical=physical,
            virtual=virtual,
            horizon=inputs.horizon,
            dimension=initial_state.dimension,
            rollout_count=inputs.rollout_count,
        )

    async def step(
        self,
        input: AugmentedControlInputSequence[
            PControlInputSequenceT, VControlInputSequenceT, T, int
        ],
        state: AugmentedState[PInStateT, VInStateT, D_x],
    ) -> AStateT:
        physical, virtual = await asyncio.gather(
            self.physical.step(input=input.physical, state=state.physical),
            self.virtual.step(input=input.virtual, state=state.virtual),
        )

        return self.state.of(
            physical=physical, virtual=virtual, dimension=state.dimension
        )


@dataclass(kw_only=True, frozen=True)
class AugmentedSampler[
    PhysicalSequenceT: ControlInputSequence,
    PhysicalBatchT: ControlInputBatch,
    VirtualSequenceT: ControlInputSequence,
    VirtualBatchT: ControlInputBatch,
    ABatchT: AugmentedControlInputBatch,
](
    Sampler[
        AugmentedControlInputSequence[PhysicalSequenceT, VirtualSequenceT],
        AugmentedControlInputBatch[PhysicalBatchT, VirtualBatchT],
    ]
):
    physical: Sampler[PhysicalSequenceT, PhysicalBatchT]
    virtual: Sampler[VirtualSequenceT, VirtualBatchT]
    batch: AugmentedControlInputBatchCreator[PhysicalBatchT, VirtualBatchT, ABatchT]

    @staticmethod
    def of[
        PS: ControlInputSequence,
        PB: ControlInputBatch,
        VS: ControlInputSequence,
        VB: ControlInputBatch,
        AB: AugmentedControlInputBatch,
    ](
        *,
        physical: Sampler[PS, PB],
        virtual: Sampler[VS, VB],
        batch: AugmentedControlInputBatchCreator[PB, VB, AB],
    ) -> "AugmentedSampler[PS, PB, VS, VB, AB]":
        return AugmentedSampler(physical=physical, virtual=virtual, batch=batch)

    def __post_init__(self) -> None:
        assert self.physical.rollout_count == self.virtual.rollout_count, (
            f"Rollout count mismatch in {self.__class__.__name__}: "
            f"got {self.physical.rollout_count} (physical) and {self.virtual.rollout_count} (virtual)"
        )

    def sample(
        self,
        *,
        around: AugmentedControlInputSequence[PhysicalSequenceT, VirtualSequenceT],
    ) -> ABatchT:
        physical_batch = self.physical.sample(around=around.physical)
        virtual_batch = self.virtual.sample(around=around.virtual)

        return self.batch.of(
            physical=physical_batch,
            virtual=virtual_batch,
            horizon=around.horizon,
            dimension=around.dimension,
            rollout_count=self.rollout_count,
        )

    @property
    def rollout_count(self) -> int:
        return self.physical.rollout_count


class HasPhysical[P](Protocol):
    @property
    def physical(self) -> P:
        """Returns the physical part."""
        ...


class HasVirtual[V](Protocol):
    @property
    def virtual(self) -> V:
        """Returns the virtual part."""
        ...


class extract:
    @staticmethod
    def from_physical[R, P](
        extract: Callable[[P], R],
    ) -> Callable[[HasPhysical[P]], R]:
        return lambda it: extract(it.physical)

    @staticmethod
    def from_virtual[R, V](extract: Callable[[V], R]) -> Callable[[HasVirtual[V]], R]:
        return lambda it: extract(it.virtual)
