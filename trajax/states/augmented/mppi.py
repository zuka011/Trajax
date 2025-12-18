from typing import Protocol

from trajax.mppi import (
    Mppi,
    DynamicalModel,
    Sampler,
    State,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
    Costs,
    CostFunction,
    NumPyState,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyCosts,
    NumPyMppi,
    NumPyUpdateFunction,
    NumPyPaddingFunction,
    NumPyFilterFunction,
)
from trajax.states.augmented.common import (
    AugmentedState,
    AugmentedStateBatch,
    AugmentedControlInputSequence,
    AugmentedControlInputBatch,
    AugmentedModel,
    AugmentedSampler,
    AugmentedStateCreator,
    AugmentedStateBatchCreator,
    AugmentedControlInputBatchCreator,
)
from trajax.states.augmented.basic import NumPyAugmentedControlInputSequence


class MppiCreator[
    StateT: State,
    StateBatchT: StateBatch,
    InputT: ControlInputSequence,
    InputBatchT: ControlInputBatch,
    CostsT: Costs,
](Protocol):
    def __call__(
        self,
        model: DynamicalModel[StateT, StateBatchT, InputT, InputBatchT],
        sampler: Sampler[InputT, InputBatchT],
        cost: CostFunction[InputBatchT, StateBatchT, CostsT],
    ) -> Mppi[StateT, InputT]:
        """Creates an MPPI planner from the specified components."""
        ...


class AugmentedMppi:
    @staticmethod
    def create[
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
        ACIS: AugmentedControlInputSequence,
        C: Costs,
    ](
        *,
        mppi: MppiCreator[AS, ASB, ACIS, AugmentedControlInputBatch[PCIB, VCIB], C],
        models: tuple[
            DynamicalModel[PS, PSB, PCIS, PCIB], DynamicalModel[VS, VSB, VCIS, VCIB]
        ],
        samplers: tuple[Sampler[PCIS, PCIB], Sampler[VCIS, VCIB]],
        cost: CostFunction[AugmentedControlInputBatch[PCIB, VCIB], ASB, C],
        state: AugmentedStateCreator[PS, VS, AS],
        state_batch: AugmentedStateBatchCreator[PSB, VSB, ASB],
        input_batch: AugmentedControlInputBatchCreator[
            PCIB, VCIB, AugmentedControlInputBatch[PCIB, VCIB]
        ],
    ) -> tuple[
        Mppi[AS, ACIS],
        AugmentedModel[PS, PSB, PCIS, PCIB, VS, VSB, VCIS, VCIB, AS, ASB],
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
        PCIS: NumPyControlInputSequence,
        PCIB: NumPyControlInputBatch,
        VS: NumPyState,
        VSB: NumPyStateBatch,
        VCIS: NumPyControlInputSequence,
        VCIB: NumPyControlInputBatch,
        AS: AugmentedState,
        ASB: AugmentedStateBatch,
        ACIS: NumPyAugmentedControlInputSequence,
        C: NumPyCosts,
    ](
        *,
        planning_interval: int = 1,
        models: tuple[
            DynamicalModel[PS, PSB, PCIS, PCIB], DynamicalModel[VS, VSB, VCIS, VCIB]
        ],
        samplers: tuple[Sampler[PCIS, PCIB], Sampler[VCIS, VCIB]],
        cost: CostFunction[AugmentedControlInputBatch[PCIB, VCIB], ASB, C],
        state: AugmentedStateCreator[PS, VS, AS],
        state_batch: AugmentedStateBatchCreator[PSB, VSB, ASB],
        input_batch: AugmentedControlInputBatchCreator[
            PCIB, VCIB, AugmentedControlInputBatch[PCIB, VCIB]
        ],
        update_function: NumPyUpdateFunction[ACIS] | None = None,
        padding_function: NumPyPaddingFunction[ACIS, NumPyAugmentedControlInputSequence]
        | None = None,
        filter_function: NumPyFilterFunction[ACIS] | None = None,
    ) -> tuple[
        Mppi[AS, ACIS],
        AugmentedModel[PS, PSB, PCIS, PCIB, VS, VSB, VCIS, VCIB, AS, ASB],
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
