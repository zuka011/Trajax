from typing import Final, Sequence
from dataclasses import dataclass

from trajax import (
    Mppi,
    AugmentedModel,
    AugmentedSampler,
    ContouringCost,
    Trajectory,
    mppi,
    model,
    sampler,
    costs,
    trajectory,
    types,
    extract,
)

from numtypes import array
from jaxtyping import Array as JaxArray, Float

import jax.numpy as jnp
import jax.random as jrandom

type PhysicalState = types.jax.bicycle.State
type PhysicalStateBatch = types.jax.bicycle.StateBatch
type PhysicalInputSequence = types.jax.bicycle.ControlInputSequence
type PhysicalInputBatch = types.jax.bicycle.ControlInputBatch
type VirtualState = types.jax.simple.State
type VirtualStateBatch = types.jax.simple.StateBatch
type VirtualInputSequence = types.jax.simple.ControlInputSequence
type VirtualInputBatch = types.jax.simple.ControlInputBatch
type MpccState = types.jax.augmented.State[PhysicalState, VirtualState]
type MpccStateBatch = types.jax.augmented.StateBatch[
    PhysicalStateBatch, VirtualStateBatch
]
type MpccInputSequence = types.jax.augmented.ControlInputSequence[
    PhysicalInputSequence, VirtualInputSequence
]
type MpccInputBatch = types.jax.augmented.ControlInputBatch[
    PhysicalInputBatch, VirtualInputBatch
]
type Model = AugmentedModel[
    PhysicalState,
    PhysicalStateBatch,
    PhysicalInputSequence,
    PhysicalInputBatch,
    VirtualState,
    VirtualStateBatch,
    VirtualInputSequence,
    VirtualInputBatch,
    MpccState,
    MpccStateBatch,
]
type Sampler = AugmentedSampler[
    PhysicalInputSequence,
    PhysicalInputBatch,
    VirtualInputSequence,
    VirtualInputBatch,
    MpccInputBatch,
]
type Planner = Mppi[MpccState, MpccInputSequence]


def path_parameter(states: VirtualStateBatch) -> types.jax.PathParameters:
    return types.jax.path_parameters(states.array[:, 0, :])


def path_velocity(inputs: VirtualInputBatch) -> Float[JaxArray, "T M"]:
    return inputs.array[:, 0, :]


def position(states: PhysicalStateBatch) -> types.jax.Positions:
    return types.jax.positions(x=states.positions.x_array, y=states.positions.y_array)


REFERENCE: Final = trajectory.jax.waypoints(
    points=array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 5.0],
            [25.0, 15.0],
            [20.0, 25.0],
            [10.0, 25.0],
            [5.0, 15.0],
            [10.0, 5.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 5.0],
            [50.0, 5.0],
            [60.0, 0.0],
            [70.0, 0.0],
        ],
        shape=(14, 2),
    ),
    path_length=120.0,
)


@dataclass(kw_only=True, frozen=True)
class JaxMpccPlannerConfiguration:
    reference: Trajectory
    planner: Planner
    model: AugmentedModel
    contouring_cost: ContouringCost
    wheelbase: float

    @staticmethod
    def stack_states(states: Sequence[MpccState]) -> MpccStateBatch:
        return types.jax.augmented.state_batch.of(
            physical=types.jax.bicycle.state_batch.of_states(
                [it.physical for it in states]
            ),
            virtual=types.jax.simple.state_batch.of_states(
                [it.virtual for it in states]
            ),
        )

    @staticmethod
    def zero_inputs(horizon: int) -> MpccInputBatch:
        return types.jax.augmented.control_input_batch.of(
            physical=types.jax.bicycle.control_input_batch.zero(horizon=horizon),
            virtual=types.jax.simple.control_input_batch.zero(
                horizon=horizon, dimension=1
            ),
        )

    @property
    def initial_state(self) -> MpccState:
        return types.jax.augmented.state.of(
            physical=types.jax.bicycle.state(jnp.array([0.0, 0.0, 0.0, 0.0])),
            virtual=types.jax.simple.state(jnp.array([0.0])),
        )

    @property
    def nominal_input(self) -> MpccInputSequence:
        return types.jax.augmented.control_input_sequence.of(
            physical=types.jax.bicycle.control_input_sequence.zeroes(
                horizon=(horizon := 30)
            ),
            virtual=types.jax.simple.control_input_sequence.zeroes(
                horizon=horizon, dimension=1
            ),
        )


class configure:
    @staticmethod
    def jax_mpcc_planner_from_base(
        reference: Trajectory = REFERENCE,
    ) -> JaxMpccPlannerConfiguration:
        planner: Planner = mppi.jax.base(
            model=(
                augmented_model := AugmentedModel.of(
                    physical=model.jax.kinematic_bicycle(
                        time_step_size=(dt := 0.1),
                        wheelbase=(L := 2.5),
                        speed_limits=(0.0, 15.0),
                        steering_limits=(-0.5, 0.5),
                        acceleration_limits=(-3.0, 3.0),
                    ),
                    virtual=model.jax.integrator(
                        time_step_size=dt,
                        state_limits=(0, reference.path_length),
                        velocity_limits=(0, 15),
                    ),
                    state=types.jax.augmented.state,
                    batch=types.jax.augmented.state_batch,
                )
            ),
            cost_function=costs.jax.combined(
                contouring_cost := costs.jax.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_extractor := extract.from_virtual(path_parameter)
                    ),
                    position_extractor=(
                        position_extractor := extract.from_physical(position)
                    ),
                    weight=50.0,
                ),
                costs.jax.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=10.0,
                ),
                costs.jax.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(path_velocity),
                    time_step_size=dt,
                    weight=250.0,
                ),
                costs.jax.comfort.control_smoothing(weights=jnp.array([2.0, 5.0, 5.0])),
            ),
            sampler=AugmentedSampler.of(
                physical=sampler.jax.gaussian(
                    standard_deviation=jnp.array([1.0, 1.0]),
                    rollout_count=(rollout_count := 512),
                    to_batch=types.jax.bicycle.control_input_batch.create,
                    key=jrandom.PRNGKey(42),
                ),
                virtual=sampler.jax.gaussian(
                    standard_deviation=jnp.array([1.0]),
                    rollout_count=rollout_count,
                    to_batch=types.jax.simple.control_input_batch.create,
                    key=jrandom.PRNGKey(43),
                ),
                batch=types.jax.augmented.control_input_batch,
            ),
        )

        return JaxMpccPlannerConfiguration(
            reference=reference,
            planner=planner,
            model=augmented_model,
            contouring_cost=contouring_cost,
            wheelbase=L,
        )

    @staticmethod
    def jax_mpcc_planner_from_augmented(
        reference: Trajectory = REFERENCE,
    ) -> JaxMpccPlannerConfiguration:
        planner, augmented_model = mppi.jax.augmented(
            models=(
                model.jax.kinematic_bicycle(
                    time_step_size=(dt := 0.1),
                    wheelbase=(L := 2.5),
                    speed_limits=(0.0, 15.0),
                    steering_limits=(-0.5, 0.5),
                    acceleration_limits=(-3.0, 3.0),
                ),
                model.jax.integrator(
                    time_step_size=dt,
                    state_limits=(0, reference.path_length),
                    velocity_limits=(0, 15),
                ),
            ),
            samplers=(
                sampler.jax.gaussian(
                    standard_deviation=jnp.array([1.0, 1.0]),
                    rollout_count=(rollout_count := 512),
                    to_batch=types.jax.bicycle.control_input_batch.create,
                    key=jrandom.PRNGKey(42),
                ),
                sampler.jax.gaussian(
                    standard_deviation=jnp.array([1.0]),
                    rollout_count=rollout_count,
                    to_batch=types.jax.simple.control_input_batch.create,
                    key=jrandom.PRNGKey(43),
                ),
            ),
            cost=costs.jax.combined(
                contouring_cost := costs.jax.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_extractor := extract.from_virtual(path_parameter)
                    ),
                    position_extractor=(
                        position_extractor := extract.from_physical(position)
                    ),
                    weight=50.0,
                ),
                costs.jax.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=10.0,
                ),
                costs.jax.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(path_velocity),
                    time_step_size=dt,
                    weight=250.0,
                ),
                costs.jax.comfort.control_smoothing(weights=jnp.array([2.0, 5.0, 5.0])),
            ),
            state=types.jax.augmented.state,
            state_batch=types.jax.augmented.state_batch,
            input_batch=types.jax.augmented.control_input_batch,
        )

        return JaxMpccPlannerConfiguration(
            reference=reference,
            planner=planner,
            model=augmented_model,
            contouring_cost=contouring_cost,
            wheelbase=L,
        )
