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


class AugmentedState[PhysicalT: State, VirtualT: State](State, Protocol):
    @property
    def physical(self) -> PhysicalT:
        """Returns the physical part of the augmented state."""
        ...

    @property
    def virtual(self) -> VirtualT:
        """Returns the virtual part of the augmented state."""
        ...


class AugmentedStateBatch[P: StateBatch, V: StateBatch](StateBatch, Protocol):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented state batch."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented state batch."""
        ...


class AugmentedControlInputSequence[P: ControlInputSequence, V: ControlInputSequence](
    ControlInputSequence, Protocol
):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented control input sequence."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented control input sequence."""
        ...


class AugmentedControlInputBatch[P: ControlInputBatch, V: ControlInputBatch](
    ControlInputBatch, Protocol
):
    @property
    def physical(self) -> P:
        """Returns the physical part of the augmented control input batch."""
        ...

    @property
    def virtual(self) -> V:
        """Returns the virtual part of the augmented control input batch."""
        ...


class AugmentedStateCreator[P: State, V: State, A: AugmentedState](Protocol):
    def of(self, *, physical: P, virtual: V) -> A:
        """Creates an augmented state from physical and virtual parts."""
        ...


class AugmentedStateBatchCreator[P: StateBatch, V: StateBatch, A: AugmentedStateBatch](
    Protocol
):
    def of(self, *, physical: P, virtual: V) -> A:
        """Creates an augmented state batch from physical and virtual parts."""
        ...


class AugmentedControlInputBatchCreator[
    P: ControlInputBatch,
    V: ControlInputBatch,
    A: AugmentedControlInputBatch,
](Protocol):
    def of(self, *, physical: P, virtual: V) -> A:
        """Creates an augmented control input batch from physical and virtual parts."""
        ...


@dataclass(kw_only=True, frozen=True)
class AugmentedModel[
    PStateT: State,
    PStateBatchT: StateBatch,
    PControlInputSequenceT: ControlInputSequence,
    PControlInputBatchT: ControlInputBatch,
    VStateT: State,
    VStateBatchT: StateBatch,
    VControlInputSequenceT: ControlInputSequence,
    VControlInputBatchT: ControlInputBatch,
    AStateT: AugmentedState,
    AStateBatchT: AugmentedStateBatch,
](
    DynamicalModel[
        AStateT,
        AStateBatchT,
        AugmentedControlInputSequence[PControlInputSequenceT, VControlInputSequenceT],
        AugmentedControlInputBatch[PControlInputBatchT, VControlInputBatchT],
    ]
):
    physical: DynamicalModel[
        PStateT, PStateBatchT, PControlInputSequenceT, PControlInputBatchT
    ]
    virtual: DynamicalModel[
        VStateT, VStateBatchT, VControlInputSequenceT, VControlInputBatchT
    ]

    state: AugmentedStateCreator[PStateT, VStateT, AStateT]
    batch: AugmentedStateBatchCreator[PStateBatchT, VStateBatchT, AStateBatchT]

    @staticmethod
    def of[
        PS: State,
        PSB: StateBatch,
        PCIS: ControlInputSequence,
        PCIB: ControlInputBatch,
        VS: State,
        VSB: StateBatch,
        VCIS: ControlInputSequence,
        VCIB: ControlInputBatch,
        AS: AugmentedState,
        ASB: AugmentedStateBatch,
    ](
        *,
        physical: DynamicalModel[PS, PSB, PCIS, PCIB],
        virtual: DynamicalModel[VS, VSB, VCIS, VCIB],
        state: AugmentedStateCreator[PS, VS, AS],
        batch: AugmentedStateBatchCreator[PSB, VSB, ASB],
    ) -> "AugmentedModel[PS, PSB, PCIS, PCIB, VS, VSB, VCIS, VCIB, AS, ASB]":
        return AugmentedModel(
            physical=physical, virtual=virtual, state=state, batch=batch
        )

    async def simulate(
        self,
        inputs: AugmentedControlInputBatch[PControlInputBatchT, VControlInputBatchT],
        initial_state: AStateT,
    ) -> AStateBatchT:
        physical, virtual = await asyncio.gather(
            self.physical.simulate(
                inputs=inputs.physical, initial_state=initial_state.physical
            ),
            self.virtual.simulate(
                inputs=inputs.virtual, initial_state=initial_state.virtual
            ),
        )

        return self.batch.of(physical=physical, virtual=virtual)

    async def step(
        self,
        input: AugmentedControlInputSequence[
            PControlInputSequenceT, VControlInputSequenceT
        ],
        state: AStateT,
    ) -> AStateT:
        physical, virtual = await asyncio.gather(
            self.physical.step(input=input.physical, state=state.physical),
            self.virtual.step(input=input.virtual, state=state.virtual),
        )

        return self.state.of(physical=physical, virtual=virtual)


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

        return self.batch.of(physical=physical_batch, virtual=virtual_batch)

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
