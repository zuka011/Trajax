from typing import Protocol, NamedTuple

from trajax.types import (
    Mppi,
    DynamicalModel,
    Sampler,
    CostFunction,
    NumPyState,
    NumPyStateSequence,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyDynamicalModel,
    NumPyCosts,
    NumPyCostFunction,
    NumPySampler,
    NumPyUpdateFunction,
    NumPyPaddingFunction,
    NumPyFilterFunction,
    JaxState,
    JaxStateSequence,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxDynamicalModel,
    JaxCosts,
    JaxCostFunction,
    JaxSampler,
    JaxUpdateFunction,
    JaxPaddingFunction,
    JaxFilterFunction,
    AugmentedState,
    AugmentedStateSequence,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
    AugmentedStateCreator,
    AugmentedStateSequenceCreator,
    AugmentedStateBatchCreator,
    AugmentedControlInputBatchCreator,
)
from trajax.mppi import NumPyWeights, NumPyMppi, JaxWeights, JaxMppi
from trajax.states.augmented.basic import (
    NumPyAugmentedState,
    NumPyAugmentedStateSequence,
    NumPyAugmentedStateBatch,
    NumPyAugmentedControlInputSequence,
    NumPyAugmentedControlInputBatch,
)
from trajax.states.augmented.accelerated import (
    JaxAugmentedState,
    JaxAugmentedStateSequence,
    JaxAugmentedStateBatch,
    JaxAugmentedControlInputSequence,
    JaxAugmentedControlInputBatch,
)
from trajax.states.augmented.common import AugmentedModel, AugmentedSampler


class MppiCreator[
    StateT,
    StateSequenceT,
    StateBatchT,
    InputSequenceT,
    WeightsT,
    InputBatchT,
    CostsT,
](Protocol):
    def __call__(
        self,
        model: DynamicalModel[
            StateT, StateSequenceT, StateBatchT, InputSequenceT, InputBatchT
        ],
        sampler: Sampler[InputSequenceT, InputBatchT],
        cost: CostFunction[InputBatchT, StateBatchT, CostsT],
    ) -> Mppi[StateT, InputSequenceT, WeightsT]:
        """Creates an MPPI planner from the specified components."""
        ...


class AugmentedMppiSetup[MppiT, ModelT](NamedTuple):
    mppi: MppiT
    model: ModelT


class AugmentedMppi:
    @staticmethod
    def create[
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
        AIS: AugmentedControlInputSequence,
        AIB: AugmentedControlInputBatch,
        W,
        C,
    ](
        *,
        mppi: MppiCreator[AS, ASS, ASB, AIS, W, AIB, C],
        models: tuple[
            DynamicalModel[PS, PSS, PSB, PIS, PIB],
            DynamicalModel[VS, VSS, VSB, VIS, VIB],
        ],
        samplers: tuple[Sampler[PIS, PIB], Sampler[VIS, VIB]],
        cost: CostFunction[AIB, ASB, C],
        state: AugmentedStateCreator[PS, VS, AS],
        state_sequence: AugmentedStateSequenceCreator[PSS, VSS, ASS],
        state_batch: AugmentedStateBatchCreator[PSB, VSB, ASB],
        input_batch: AugmentedControlInputBatchCreator[PIB, VIB, AIB],
    ) -> AugmentedMppiSetup[
        Mppi[AS, AIS, W],
        AugmentedModel[PS, PSS, PSB, PIS, PIB, VS, VSS, VSB, VIS, VIB, AS, ASS, ASB],
    ]:
        return AugmentedMppiSetup(
            mppi=mppi(
                model=(
                    model := AugmentedModel.of(
                        physical=models[0],
                        virtual=models[1],
                        state=state,
                        sequence=state_sequence,
                        batch=state_batch,
                    )
                ),
                sampler=AugmentedSampler.of(
                    physical=samplers[0], virtual=samplers[1], batch=input_batch
                ),
                cost=cost,
            ),
            model=model,
        )


class NumPyAugmentedMppi:
    @staticmethod
    def create[
        PS: NumPyState,
        PSS: NumPyStateSequence,
        PSB: NumPyStateBatch,
        PCS: NumPyControlInputSequence,
        PCB: NumPyControlInputBatch,
        VS: NumPyState,
        VSS: NumPyStateSequence,
        VSB: NumPyStateBatch,
        VCS: NumPyControlInputSequence,
        VCB: NumPyControlInputBatch,
        AS: NumPyAugmentedState,
        ASS: NumPyAugmentedStateSequence,
        ASB: NumPyAugmentedStateBatch,
        ACS: NumPyAugmentedControlInputSequence,
        C: NumPyCosts,
    ](
        *,
        planning_interval: int = 1,
        models: tuple[
            NumPyDynamicalModel[PS, PSS, PSB, PCS, PCB],
            NumPyDynamicalModel[VS, VSS, VSB, VCS, VCB],
        ],
        samplers: tuple[NumPySampler[PCS, PCB], NumPySampler[VCS, VCB]],
        cost: NumPyCostFunction[NumPyAugmentedControlInputBatch[PCB, VCB], ASB, C],
        state: AugmentedStateCreator[PS, VS, AS],
        state_sequence: AugmentedStateSequenceCreator[PSS, VSS, ASS],
        state_batch: AugmentedStateBatchCreator[PSB, VSB, ASB],
        input_batch: AugmentedControlInputBatchCreator[
            PCB, VCB, NumPyAugmentedControlInputBatch[PCB, VCB]
        ],
        update_function: NumPyUpdateFunction[ACS] | None = None,
        padding_function: NumPyPaddingFunction[ACS, NumPyAugmentedControlInputSequence]
        | None = None,
        filter_function: NumPyFilterFunction[ACS] | None = None,
    ) -> AugmentedMppiSetup[
        Mppi[AS, ACS, NumPyWeights],
        AugmentedModel[PS, PSS, PSB, PCS, PCB, VS, VSS, VSB, VCS, VCB, AS, ASS, ASB],
    ]:
        return AugmentedMppi.create(
            mppi=lambda model, sampler, cost: NumPyMppi.create(
                planning_interval=planning_interval,
                model=model,
                cost_function=cost,
                sampler=sampler,
                update_function=update_function,
                padding_function=padding_function,
                filter_function=filter_function,
            ),
            models=models,
            samplers=samplers,
            cost=cost,
            state=state,
            state_sequence=state_sequence,
            state_batch=state_batch,
            input_batch=input_batch,
        )


class JaxAugmentedMppi:
    @staticmethod
    def create[
        PS: JaxState,
        PSS: JaxStateSequence,
        PSB: JaxStateBatch,
        PCS: JaxControlInputSequence,
        PCB: JaxControlInputBatch,
        VS: JaxState,
        VSS: JaxStateSequence,
        VSB: JaxStateBatch,
        VCS: JaxControlInputSequence,
        VCB: JaxControlInputBatch,
        AS: JaxAugmentedState,
        ASS: JaxAugmentedStateSequence,
        ASB: JaxAugmentedStateBatch,
        ACS: JaxAugmentedControlInputSequence,
        C: JaxCosts,
    ](
        *,
        planning_interval: int = 1,
        models: tuple[
            JaxDynamicalModel[PS, PSS, PSB, PCS, PCB],
            JaxDynamicalModel[VS, VSS, VSB, VCS, VCB],
        ],
        samplers: tuple[JaxSampler[PCS, PCB], JaxSampler[VCS, VCB]],
        cost: JaxCostFunction[JaxAugmentedControlInputBatch[PCB, VCB], ASB, C],
        state: AugmentedStateCreator[PS, VS, AS],
        state_sequence: AugmentedStateSequenceCreator[PSS, VSS, ASS],
        state_batch: AugmentedStateBatchCreator[PSB, VSB, ASB],
        input_batch: AugmentedControlInputBatchCreator[
            PCB, VCB, JaxAugmentedControlInputBatch[PCB, VCB]
        ],
        update_function: JaxUpdateFunction[ACS] | None = None,
        padding_function: JaxPaddingFunction[ACS, JaxAugmentedControlInputSequence]
        | None = None,
        filter_function: JaxFilterFunction[ACS] | None = None,
    ) -> AugmentedMppiSetup[
        Mppi[AS, ACS, JaxWeights],
        AugmentedModel[PS, PSS, PSB, PCS, PCB, VS, VSS, VSB, VCS, VCB, AS, ASS, ASB],
    ]:
        return AugmentedMppi.create(
            mppi=lambda model, sampler, cost: JaxMppi.create(
                planning_interval=planning_interval,
                model=model,
                cost_function=cost,
                sampler=sampler,
                update_function=update_function,
                padding_function=padding_function,
                filter_function=filter_function,
            ),
            models=models,
            samplers=samplers,
            cost=cost,
            state=state,
            state_sequence=state_sequence,
            state_batch=state_batch,
            input_batch=input_batch,
        )
