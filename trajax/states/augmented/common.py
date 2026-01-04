from typing import Callable
from dataclasses import dataclass

from trajax.types import (
    DynamicalModel,
    Sampler,
    AugmentedState,
    AugmentedStateSequence,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
    AugmentedStateCreator,
    AugmentedStateSequenceCreator,
    AugmentedStateBatchCreator,
    AugmentedControlInputBatchCreator,
    HasPhysical,
    HasVirtual,
)


@dataclass(kw_only=True, frozen=True)
class AugmentedModel[
    PStateT,
    PStateSequenceT,
    PStateBatchT,
    PInputSequenceT,
    PInputBatchT,
    VStateT,
    VStateSequenceT,
    VStateBatchT,
    VInputSequenceT,
    VInputBatchT,
    AStateT: AugmentedState,
    AStateSequenceT: AugmentedStateSequence,
    AStateBatchT: AugmentedStateBatch,
](
    DynamicalModel[
        AStateT,
        AStateSequenceT,
        AStateBatchT,
        AugmentedControlInputSequence[PInputSequenceT, VInputSequenceT],
        AugmentedControlInputBatch[PInputBatchT, VInputBatchT],
    ]
):
    physical: DynamicalModel[
        PStateT, PStateSequenceT, PStateBatchT, PInputSequenceT, PInputBatchT
    ]
    virtual: DynamicalModel[
        VStateT, VStateSequenceT, VStateBatchT, VInputSequenceT, VInputBatchT
    ]

    state: AugmentedStateCreator[PStateT, VStateT, AStateT]
    sequence: AugmentedStateSequenceCreator[
        PStateSequenceT, VStateSequenceT, AStateSequenceT
    ]
    batch: AugmentedStateBatchCreator[PStateBatchT, VStateBatchT, AStateBatchT]

    @staticmethod
    def of[
        PS,
        PSS,
        PSB,
        PIS,
        PIB,
        VS,
        VSS,
        VSB,
        VIS,
        VIB,
        AS: AugmentedState,
        ASS: AugmentedStateSequence,
        ASB: AugmentedStateBatch,
    ](
        *,
        physical: DynamicalModel[PS, PSS, PSB, PIS, PIB],
        virtual: DynamicalModel[VS, VSS, VSB, VIS, VIB],
        state: AugmentedStateCreator[PS, VS, AS],
        sequence: AugmentedStateSequenceCreator[PSS, VSS, ASS],
        batch: AugmentedStateBatchCreator[PSB, VSB, ASB],
    ) -> "AugmentedModel[PS, PSS, PSB, PIS, PIB, VS, VSS, VSB, VIS, VIB, AS, ASS, ASB]":
        return AugmentedModel(
            physical=physical,
            virtual=virtual,
            state=state,
            sequence=sequence,
            batch=batch,
        )

    def __post_init__(self) -> None:
        assert self.physical.time_step_size == self.virtual.time_step_size, (
            f"Physical and virtual models have different time step sizes: "
            f"{self.physical.time_step_size} (physical) vs {self.virtual.time_step_size} (virtual)"
        )

    def simulate(
        self,
        inputs: AugmentedControlInputBatch[PInputBatchT, VInputBatchT],
        initial_state: AStateT,
    ) -> AStateBatchT:
        physical, virtual = (
            self.physical.simulate(
                inputs=inputs.physical, initial_state=initial_state.physical
            ),
            self.virtual.simulate(
                inputs=inputs.virtual, initial_state=initial_state.virtual
            ),
        )

        return self.batch.of(physical=physical, virtual=virtual)

    def step(
        self,
        inputs: AugmentedControlInputSequence[PInputSequenceT, VInputSequenceT],
        state: AStateT,
    ) -> AStateT:
        physical, virtual = (
            self.physical.step(inputs=inputs.physical, state=state.physical),
            self.virtual.step(inputs=inputs.virtual, state=state.virtual),
        )

        return self.state.of(physical=physical, virtual=virtual)

    def forward(
        self,
        inputs: AugmentedControlInputSequence[PInputSequenceT, VInputSequenceT],
        state: AStateT,
    ) -> AStateSequenceT:
        physical, virtual = (
            self.physical.forward(inputs=inputs.physical, state=state.physical),
            self.virtual.forward(inputs=inputs.virtual, state=state.virtual),
        )

        return self.sequence.of(physical=physical, virtual=virtual)

    @property
    def time_step_size(self) -> float:
        return self.physical.time_step_size


@dataclass(kw_only=True, frozen=True)
class AugmentedSampler[
    PInputSequenceT,
    PInputBatchT,
    VInputSequenceT,
    VInputBatchT,
    ABatchT: AugmentedControlInputBatch,
](Sampler[AugmentedControlInputSequence[PInputSequenceT, VInputSequenceT], ABatchT]):
    physical: Sampler[PInputSequenceT, PInputBatchT]
    virtual: Sampler[VInputSequenceT, VInputBatchT]
    batch: AugmentedControlInputBatchCreator[PInputBatchT, VInputBatchT, ABatchT]

    @staticmethod
    def of[PS, PB, VS, VB, AB: AugmentedControlInputBatch](
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
        around: AugmentedControlInputSequence[PInputSequenceT, VInputSequenceT],
    ) -> ABatchT:
        physical_batch = self.physical.sample(around=around.physical)
        virtual_batch = self.virtual.sample(around=around.virtual)

        return self.batch.of(physical=physical_batch, virtual=virtual_batch)

    @property
    def rollout_count(self) -> int:
        return self.physical.rollout_count


class extract:
    @staticmethod
    def from_physical[R, P](
        extract: Callable[[P], R],
    ) -> Callable[[HasPhysical[P]], R]:
        return lambda it: extract(it.physical)

    @staticmethod
    def from_virtual[R, V](extract: Callable[[V], R]) -> Callable[[HasVirtual[V]], R]:
        return lambda it: extract(it.virtual)
