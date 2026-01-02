from typing import Protocol

from trajax.types import (
    Mppi,
    DynamicalModel,
    Sampler,
    CostFunction,
    NumPyState,
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
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
    AugmentedStateCreator,
    AugmentedStateBatchCreator,
    AugmentedControlInputBatchCreator,
)
from trajax.mppi import NumPyWeights, NumPyMppi, JaxWeights, JaxMppi
from trajax.states.augmented.basic import (
    NumPyAugmentedState,
    NumPyAugmentedStateBatch,
    NumPyAugmentedControlInputSequence,
    NumPyAugmentedControlInputBatch,
)
from trajax.states.augmented.accelerated import (
    JaxAugmentedState,
    JaxAugmentedStateBatch,
    JaxAugmentedControlInputSequence,
    JaxAugmentedControlInputBatch,
)
from trajax.states.augmented.common import AugmentedModel, AugmentedSampler


class MppiCreator[StateT, StateBatchT, InputSequenceT, WeightsT, InputBatchT, CostsT](
    Protocol
):
    def __call__(
        self,
        model: DynamicalModel[StateT, StateBatchT, InputSequenceT, InputBatchT],
        sampler: Sampler[InputSequenceT, InputBatchT],
        cost: CostFunction[InputBatchT, StateBatchT, CostsT],
    ) -> Mppi[StateT, InputSequenceT, WeightsT]:
        """Creates an MPPI planner from the specified components."""
        ...


class AugmentedMppi:
    @staticmethod
    def create[
        PS,
        PSB,
        PIS,
        PIB,
        VS,
        VSB,
        VIS,
        VIB,
        AS: AugmentedState,
        ASB: AugmentedStateBatch,
        AIS: AugmentedControlInputSequence,
        AIB: AugmentedControlInputBatch,
        W,
        C,
    ](
        *,
        mppi: MppiCreator[AS, ASB, AIS, W, AIB, C],
        models: tuple[
            DynamicalModel[PS, PSB, PIS, PIB], DynamicalModel[VS, VSB, VIS, VIB]
        ],
        samplers: tuple[Sampler[PIS, PIB], Sampler[VIS, VIB]],
        cost: CostFunction[AIB, ASB, C],
        state: AugmentedStateCreator[PS, VS, AS],
        state_batch: AugmentedStateBatchCreator[PSB, VSB, ASB],
        input_batch: AugmentedControlInputBatchCreator[PIB, VIB, AIB],
    ) -> tuple[
        Mppi[AS, AIS, W],
        AugmentedModel[PS, PSB, PIS, PIB, VS, VSB, VIS, VIB, AS, ASB],
    ]:
        return (
            mppi(
                model=(
                    model := AugmentedModel.of(
                        physical=models[0],
                        virtual=models[1],
                        state=state,
                        batch=state_batch,
                    )
                ),
                sampler=AugmentedSampler.of(
                    physical=samplers[0], virtual=samplers[1], batch=input_batch
                ),
                cost=cost,
            ),
            model,
        )


class NumPyAugmentedMppi:
    @staticmethod
    def create[
        PS: NumPyState,
        PSB: NumPyStateBatch,
        PCS: NumPyControlInputSequence,
        PCB: NumPyControlInputBatch,
        VS: NumPyState,
        VSB: NumPyStateBatch,
        VCS: NumPyControlInputSequence,
        VCB: NumPyControlInputBatch,
        AS: NumPyAugmentedState,
        ASB: NumPyAugmentedStateBatch,
        ACS: NumPyAugmentedControlInputSequence,
        C: NumPyCosts,
    ](
        *,
        planning_interval: int = 1,
        models: tuple[
            NumPyDynamicalModel[PS, PSB, PCS, PCB],
            NumPyDynamicalModel[VS, VSB, VCS, VCB],
        ],
        samplers: tuple[NumPySampler[PCS, PCB], NumPySampler[VCS, VCB]],
        cost: NumPyCostFunction[NumPyAugmentedControlInputBatch[PCB, VCB], ASB, C],
        state: AugmentedStateCreator[PS, VS, AS],
        state_batch: AugmentedStateBatchCreator[PSB, VSB, ASB],
        input_batch: AugmentedControlInputBatchCreator[
            PCB, VCB, NumPyAugmentedControlInputBatch[PCB, VCB]
        ],
        update_function: NumPyUpdateFunction[ACS] | None = None,
        padding_function: NumPyPaddingFunction[ACS, NumPyAugmentedControlInputSequence]
        | None = None,
        filter_function: NumPyFilterFunction[ACS] | None = None,
    ) -> tuple[
        Mppi[AS, ACS, NumPyWeights],
        AugmentedModel[PS, PSB, PCS, PCB, VS, VSB, VCS, VCB, AS, ASB],
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
            state_batch=state_batch,
            input_batch=input_batch,
        )


class JaxAugmentedMppi:
    @staticmethod
    def create[
        PS: JaxState,
        PSB: JaxStateBatch,
        PCS: JaxControlInputSequence,
        PCB: JaxControlInputBatch,
        VS: JaxState,
        VSB: JaxStateBatch,
        VCS: JaxControlInputSequence,
        VCB: JaxControlInputBatch,
        AS: JaxAugmentedState,
        ASB: JaxAugmentedStateBatch,
        ACS: JaxAugmentedControlInputSequence,
        C: JaxCosts,
    ](
        *,
        planning_interval: int = 1,
        models: tuple[
            JaxDynamicalModel[PS, PSB, PCS, PCB], JaxDynamicalModel[VS, VSB, VCS, VCB]
        ],
        samplers: tuple[JaxSampler[PCS, PCB], JaxSampler[VCS, VCB]],
        cost: JaxCostFunction[JaxAugmentedControlInputBatch[PCB, VCB], ASB, C],
        state: AugmentedStateCreator[PS, VS, AS],
        state_batch: AugmentedStateBatchCreator[PSB, VSB, ASB],
        input_batch: AugmentedControlInputBatchCreator[
            PCB, VCB, JaxAugmentedControlInputBatch[PCB, VCB]
        ],
        update_function: JaxUpdateFunction[ACS] | None = None,
        padding_function: JaxPaddingFunction[ACS, JaxAugmentedControlInputSequence]
        | None = None,
        filter_function: JaxFilterFunction[ACS] | None = None,
    ) -> tuple[
        Mppi[AS, ACS, JaxWeights],
        AugmentedModel[PS, PSB, PCS, PCB, VS, VSB, VCS, VCB, AS, ASB],
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
            state_batch=state_batch,
            input_batch=input_batch,
        )
