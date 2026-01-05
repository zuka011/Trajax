from typing import TypedDict

from trajax.types import (
    NumPyState,
    NumPyStateSequence,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyDynamicalModel,
    NumPySampler,
    NumPyCosts,
    NumPyCostFunction,
    NumPyUpdateFunction,
    NumPyPaddingFunction,
    NumPyFilterFunction,
    Mppi,
    Trajectory,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositionExtractor,
)
from trajax.states import (
    NumPySimpleState,
    NumPySimpleControlInputSequence,
    NumPyAugmentedState,
    NumPyAugmentedStateSequence,
    NumPyAugmentedStateBatch,
    NumPyAugmentedControlInputSequence,
    NumPyAugmentedControlInputBatch,
    NumPyAugmentedMppi,
    NumPySimpleControlInputBatch,
    AugmentedModel,
    extract,
)
from trajax.models import model as create_model
from trajax.samplers import sampler as create_sampler
from trajax.costs import costs as create_costs, NumPyContouringCost, NumPyLagCost
from trajax.mppi import NumPyWeights
from trajax.mpcc.common import MpccMppiSetup

from numtypes import Array, Dim2, D, array_1d
from deepmerge import always_merger

type NumPyMpccVirtualState = NumPySimpleState[D[1]]
type NumPyMpccAugmentedState[S: NumPyState] = NumPyAugmentedState[
    S, NumPyMpccVirtualState
]

type NumPyMpccVirtualStateSequence = NumPyStateSequence[int, D[1]]
type NumPyMpccAugmentedStateSequence[SS: NumPyStateSequence] = (
    NumPyAugmentedStateSequence[SS, NumPyMpccVirtualStateSequence]
)

type NumPyMpccVirtualStateBatch = NumPyStateBatch[int, D[1], int]
type NumPyMpccAugmentedStateBatch[SB: NumPyStateBatch] = NumPyAugmentedStateBatch[
    SB, NumPyMpccVirtualStateBatch
]

type NumPyMpccVirtualControlInputSequence = NumPySimpleControlInputSequence[int, D[1]]
type NumPyMpccAugmentedControlInputSequence[CS: NumPyControlInputSequence] = (
    NumPyAugmentedControlInputSequence[CS, NumPyMpccVirtualControlInputSequence]
)

type NumPyMpccVirtualControlInputBatch = NumPyControlInputBatch[int, D[1], int]
type NumPyMpccAugmentedControlInputBatch[CB: NumPyControlInputBatch] = (
    NumPyAugmentedControlInputBatch[CB, NumPyMpccVirtualControlInputBatch]
)

type NumPyMpccCostFunction[
    CB: NumPyControlInputBatch,
    SB: NumPyStateBatch,
    C: NumPyCosts,
] = NumPyCostFunction[
    NumPyMpccAugmentedControlInputBatch[CB], NumPyMpccAugmentedStateBatch[SB], C
]


class NumPyMpccWeightConfig:
    class Full(TypedDict):
        contouring: float
        lag: float
        progress: float

    class Partial(TypedDict, total=False):
        contouring: float
        lag: float
        progress: float


class NumPyMpccVirtualStateConfig:
    class Full(TypedDict):
        state_limits: tuple[float, float]
        velocity_limits: tuple[float, float]
        sampling_standard_deviation: float
        sampling_seed: int

    class Partial(TypedDict, total=False):
        state_limits: tuple[float, float]
        velocity_limits: tuple[float, float]
        sampling_standard_deviation: float
        sampling_seed: int


class NumPyMpccConfig:
    class Full(TypedDict):
        weights: NumPyMpccWeightConfig.Full
        virtual: NumPyMpccVirtualStateConfig.Full

    class Partial(TypedDict, total=False):
        weights: NumPyMpccWeightConfig.Partial
        virtual: NumPyMpccVirtualStateConfig.Partial


def fill_defaults(
    config: NumPyMpccConfig.Partial | None,
    *,
    reference: Trajectory[NumPyPathParameters, NumPyReferencePoints],
) -> NumPyMpccConfig.Full:
    return always_merger.merge(  # type: ignore
        NumPyMpccConfig.Full(
            {
                "weights": {
                    "contouring": 20.0,
                    "lag": 10.0,
                    "progress": 500.0,
                },
                "virtual": {
                    "state_limits": (0.0, reference.path_length),
                    "velocity_limits": (0.0, 15.0),
                    "sampling_standard_deviation": 1.0,
                    "sampling_seed": 0,
                },
            }
        ),
        config if config is not None else NumPyMpccConfig.Partial({}),
    )


class NumPyMpccMppi:
    @staticmethod
    def create[
        S: NumPyState,
        SS: NumPyStateSequence,
        SB: NumPyStateBatch,
        CS: NumPyControlInputSequence,
        CB: NumPyControlInputBatch,
        C: NumPyCosts,
    ](
        *,
        planning_interval: int = 1,
        model: NumPyDynamicalModel[S, SS, SB, CS, CB],
        sampler: NumPySampler[CS, CB],
        costs: tuple[NumPyMpccCostFunction[CB, SB, C], ...],
        reference: Trajectory[NumPyPathParameters, NumPyReferencePoints],
        position_extractor: NumPyPositionExtractor[NumPyMpccAugmentedStateBatch[SB]],
        config: NumPyMpccConfig.Partial | None = None,
        update_function: NumPyUpdateFunction[NumPyMpccAugmentedControlInputSequence[CS]]
        | None = None,
        padding_function: NumPyPaddingFunction[
            NumPyMpccAugmentedControlInputSequence[CS],
            NumPyMpccAugmentedControlInputSequence[CS],
        ]
        | None = None,
        filter_function: NumPyFilterFunction[NumPyMpccAugmentedControlInputSequence[CS]]
        | None = None,
    ) -> MpccMppiSetup[
        Mppi[
            NumPyMpccAugmentedState[S],
            NumPyMpccAugmentedControlInputSequence[CS],
            NumPyWeights,
        ],
        AugmentedModel[
            S,
            SS,
            SB,
            CS,
            CB,
            NumPyMpccVirtualState,
            NumPyMpccVirtualStateSequence,
            NumPyMpccVirtualStateBatch,
            NumPyMpccVirtualControlInputSequence,
            NumPyMpccVirtualControlInputBatch,
            NumPyMpccAugmentedState[S],
            NumPyMpccAugmentedStateSequence[SS],
            NumPyMpccAugmentedStateBatch[SB],
        ],
        NumPyContouringCost[NumPyMpccAugmentedStateBatch[SB]],
        NumPyLagCost[NumPyMpccAugmentedStateBatch[SB]],
    ]:
        full_config = fill_defaults(config, reference=reference)

        planner, augmented_model = NumPyAugmentedMppi.create(
            planning_interval=planning_interval,
            models=(
                model,
                create_model.numpy.integrator.dynamical(
                    time_step_size=model.time_step_size,
                    state_limits=full_config["virtual"]["state_limits"],
                    velocity_limits=full_config["virtual"]["velocity_limits"],
                ),
            ),
            samplers=(
                sampler,
                create_sampler.numpy.gaussian(
                    standard_deviation=array_1d(
                        (full_config["virtual"]["sampling_standard_deviation"],)
                    ),
                    rollout_count=sampler.rollout_count,
                    to_batch=NumPySimpleControlInputBatch.create,
                    seed=full_config["virtual"]["sampling_seed"],
                ),
            ),
            cost=create_costs.numpy.combined(  # type: ignore
                contouring_cost := create_costs.numpy.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_parameter_extractor := extract.from_virtual(
                            extract_path_parameters
                        )
                    ),
                    position_extractor=position_extractor,
                    weight=full_config["weights"]["contouring"],
                ),
                lag_cost := create_costs.numpy.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_parameter_extractor,
                    position_extractor=position_extractor,
                    weight=full_config["weights"]["lag"],
                ),
                create_costs.numpy.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(extract_path_velocity),
                    time_step_size=model.time_step_size,
                    weight=full_config["weights"]["progress"],
                ),
                *costs,
            ),
            state=NumPyAugmentedState,
            state_sequence=NumPyAugmentedStateSequence,
            state_batch=NumPyAugmentedStateBatch,  # type: ignore
            input_batch=NumPyAugmentedControlInputBatch,
            update_function=update_function,
            padding_function=padding_function,
            filter_function=filter_function,
        )

        return MpccMppiSetup(
            mppi=planner,  # type: ignore
            model=augmented_model,
            contouring_cost=contouring_cost,
            lag_cost=lag_cost,
        )


def extract_path_parameters(states: NumPyMpccVirtualStateBatch) -> NumPyPathParameters:
    return NumPyPathParameters(states.array[:, 0, :])


def extract_path_velocity(inputs: NumPyMpccVirtualControlInputBatch) -> Array[Dim2]:
    return inputs.array[:, 0, :]
