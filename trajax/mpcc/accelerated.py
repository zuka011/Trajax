from typing import TypedDict


from trajax.types import (
    Mppi,
    Trajectory,
    JaxState,
    JaxStateSequence,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxDynamicalModel,
    JaxSampler,
    JaxCosts,
    JaxCostFunction,
    JaxUpdateFunction,
    JaxPaddingFunction,
    JaxFilterFunction,
    JaxPathParameters,
    JaxReferencePoints,
    JaxPositionExtractor,
)
from trajax.states import (
    JaxSimpleState,
    JaxSimpleControlInputSequence,
    JaxAugmentedState,
    JaxAugmentedStateSequence,
    JaxAugmentedStateBatch,
    JaxAugmentedControlInputSequence,
    JaxAugmentedControlInputBatch,
    JaxAugmentedMppi,
    JaxSimpleControlInputBatch,
    AugmentedModel,
    extract,
)
from trajax.models import model as create_model
from trajax.samplers import sampler as create_sampler
from trajax.costs import (
    costs as create_costs,
    JaxContouringCost,
    JaxLagCost,
)
from trajax.mppi import JaxWeights
from trajax.mpcc.common import MpccMppiSetup

from numtypes import D
from jaxtyping import Float, Array as JaxArray, PRNGKeyArray
from deepmerge import always_merger

import jax.numpy as jnp
import jax.random as jrandom


type JaxMpccVirtualState = JaxSimpleState[D[1]]
type JaxMpccAugmentedState[S: JaxState] = JaxAugmentedState[S, JaxMpccVirtualState]

type JaxMpccVirtualStateSequence = JaxStateSequence[int, D[1]]
type JaxMpccAugmentedStateSequence[SS: JaxStateSequence] = JaxAugmentedStateSequence[
    SS, JaxMpccVirtualStateSequence
]

type JaxMpccVirtualStateBatch = JaxStateBatch[int, D[1], int]
type JaxMpccAugmentedStateBatch[SB: JaxStateBatch] = JaxAugmentedStateBatch[
    SB, JaxMpccVirtualStateBatch
]

type JaxMpccVirtualControlInputSequence = JaxSimpleControlInputSequence[int, D[1]]
type JaxMpccAugmentedControlInputSequence[CS: JaxControlInputSequence] = (
    JaxAugmentedControlInputSequence[CS, JaxMpccVirtualControlInputSequence]
)

type JaxMpccVirtualControlInputBatch = JaxControlInputBatch[int, D[1], int]
type JaxMpccAugmentedControlInputBatch[CB: JaxControlInputBatch] = (
    JaxAugmentedControlInputBatch[CB, JaxMpccVirtualControlInputBatch]
)

type JaxMpccCostFunction[
    CB: JaxControlInputBatch,
    SB: JaxStateBatch,
    C: JaxCosts,
] = JaxCostFunction[
    JaxMpccAugmentedControlInputBatch[CB], JaxMpccAugmentedStateBatch[SB], C
]


class JaxMpccWeightConfig:
    class Full(TypedDict):
        contouring: float
        lag: float
        progress: float

    class Partial(TypedDict, total=False):
        contouring: float
        lag: float
        progress: float


class JaxMpccVirtualStateConfig:
    class Full(TypedDict):
        state_limits: tuple[float, float]
        velocity_limits: tuple[float, float]
        sampling_standard_deviation: float
        sampling_key: PRNGKeyArray

    class Partial(TypedDict, total=False):
        state_limits: tuple[float, float]
        velocity_limits: tuple[float, float]
        sampling_standard_deviation: float
        sampling_key: PRNGKeyArray


class JaxMpccConfig:
    class Full(TypedDict):
        weights: JaxMpccWeightConfig.Full
        virtual: JaxMpccVirtualStateConfig.Full

    class Partial(TypedDict, total=False):
        weights: JaxMpccWeightConfig.Partial
        virtual: JaxMpccVirtualStateConfig.Partial


def fill_defaults(
    config: JaxMpccConfig.Partial | None,
    *,
    reference: Trajectory[JaxPathParameters, JaxReferencePoints],
) -> JaxMpccConfig.Full:
    return always_merger.merge(  # type: ignore
        JaxMpccConfig.Full(
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
                    "sampling_key": jrandom.PRNGKey(0),
                },
            }
        ),
        config if config is not None else JaxMpccConfig.Partial({}),
    )


class JaxMpccMppi:
    @staticmethod
    def create[
        S: JaxState,
        SS: JaxStateSequence,
        SB: JaxStateBatch,
        CS: JaxControlInputSequence,
        CB: JaxControlInputBatch,
        C: JaxCosts,
    ](
        *,
        planning_interval: int = 1,
        model: JaxDynamicalModel[S, SS, SB, CS, CB],
        sampler: JaxSampler[CS, CB],
        costs: tuple[JaxMpccCostFunction[CB, SB, C], ...],
        reference: Trajectory[JaxPathParameters, JaxReferencePoints],
        position_extractor: JaxPositionExtractor[JaxMpccAugmentedStateBatch[SB]],
        config: JaxMpccConfig.Partial | None = None,
        update_function: JaxUpdateFunction[JaxMpccAugmentedControlInputSequence[CS]]
        | None = None,
        padding_function: JaxPaddingFunction[
            JaxMpccAugmentedControlInputSequence[CS],
            JaxAugmentedControlInputSequence,
        ]
        | None = None,
        filter_function: JaxFilterFunction[JaxMpccAugmentedControlInputSequence[CS]]
        | None = None,
    ) -> MpccMppiSetup[
        Mppi[
            JaxMpccAugmentedState[S],
            JaxMpccAugmentedControlInputSequence[CS],
            JaxWeights,
        ],
        AugmentedModel[
            S,
            SS,
            SB,
            CS,
            CB,
            JaxMpccVirtualState,
            JaxMpccVirtualStateSequence,
            JaxMpccVirtualStateBatch,
            JaxMpccVirtualControlInputSequence,
            JaxMpccVirtualControlInputBatch,
            JaxMpccAugmentedState[S],
            JaxMpccAugmentedStateSequence[SS],
            JaxMpccAugmentedStateBatch[SB],
        ],
        JaxContouringCost[JaxMpccAugmentedStateBatch[SB]],
        JaxLagCost[JaxMpccAugmentedStateBatch[SB]],
    ]:
        full_config = fill_defaults(config, reference=reference)

        planner, augmented_model = JaxAugmentedMppi.create(
            planning_interval=planning_interval,
            models=(
                model,
                create_model.jax.integrator.dynamical(
                    time_step_size=model.time_step_size,
                    state_limits=full_config["virtual"]["state_limits"],
                    velocity_limits=full_config["virtual"]["velocity_limits"],
                ),
            ),
            samplers=(
                sampler,
                create_sampler.jax.gaussian(
                    standard_deviation=jnp.array(
                        [full_config["virtual"]["sampling_standard_deviation"]]
                    ),
                    rollout_count=sampler.rollout_count,
                    to_batch=JaxSimpleControlInputBatch.create,
                    key=full_config["virtual"]["sampling_key"],
                ),
            ),
            cost=create_costs.jax.combined(
                contouring_cost := create_costs.jax.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_parameter_extractor := extract.from_virtual(
                            extract_path_parameters
                        )
                    ),
                    position_extractor=position_extractor,
                    weight=full_config["weights"]["contouring"],
                ),
                lag_cost := create_costs.jax.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_parameter_extractor,
                    position_extractor=position_extractor,
                    weight=full_config["weights"]["lag"],
                ),
                create_costs.jax.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(extract_path_velocity),
                    time_step_size=model.time_step_size,
                    weight=full_config["weights"]["progress"],
                ),
                *costs,
            ),
            state=JaxAugmentedState,
            state_sequence=JaxAugmentedStateSequence,
            state_batch=JaxAugmentedStateBatch,  # type: ignore
            input_batch=JaxAugmentedControlInputBatch,
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


def extract_path_parameters(states: JaxMpccVirtualStateBatch) -> JaxPathParameters:
    return JaxPathParameters(states.array[:, 0, :])


def extract_path_velocity(
    inputs: JaxMpccVirtualControlInputBatch,
) -> Float[JaxArray, "T M"]:
    return inputs.array[:, 0, :]
