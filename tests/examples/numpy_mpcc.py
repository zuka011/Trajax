from typing import Final
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

import numpy as np
from numtypes import array, Array, Dim2

type PhysicalState = types.numpy.bicycle.State
type PhysicalStateBatch = types.numpy.bicycle.StateBatch
type PhysicalInputSequence = types.numpy.bicycle.ControlInputSequence
type PhysicalInputBatch = types.numpy.bicycle.ControlInputBatch
type VirtualState = types.numpy.simple.State
type VirtualStateBatch = types.numpy.simple.StateBatch
type VirtualInputSequence = types.numpy.simple.ControlInputSequence
type VirtualInputBatch = types.numpy.simple.ControlInputBatch
type MpccState = types.numpy.augmented.State[PhysicalState, VirtualState]
type MpccStateBatch = types.numpy.augmented.StateBatch[
    PhysicalStateBatch, VirtualStateBatch
]
type MpccInputSequence = types.numpy.augmented.ControlInputSequence[
    PhysicalInputSequence, VirtualInputSequence
]
type MpccInputBatch = types.numpy.augmented.ControlInputBatch[
    PhysicalInputBatch, VirtualInputBatch
]
type Planner = Mppi[MpccState, MpccInputSequence]


def path_parameter(states: VirtualStateBatch) -> types.numpy.PathParameters:
    return types.numpy.path_parameters(states.array[:, 0, :])


def path_velocity(inputs: VirtualInputBatch) -> Array[Dim2]:
    return inputs.array[:, 0, :]


def position(states: PhysicalStateBatch) -> types.numpy.Positions:
    return types.numpy.positions(x=states.positions.x(), y=states.positions.y())


REFERENCE: Final = trajectory.numpy.waypoints(
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
class NumPyMpccPlannerConfiguration:
    reference: Trajectory
    planner: Planner
    model: AugmentedModel
    contouring_cost: ContouringCost
    wheelbase: float

    @staticmethod
    def stack_states(states: list[MpccState]) -> MpccStateBatch:
        return types.numpy.augmented.state_batch.of(
            physical=types.numpy.bicycle.state_batch.of_states(
                [it.physical for it in states]
            ),
            virtual=types.numpy.simple.state_batch.of_states(
                [it.virtual for it in states]
            ),
        )

    @staticmethod
    def zero_inputs(horizon: int) -> MpccInputBatch:
        return types.numpy.augmented.control_input_batch.of(
            physical=types.numpy.bicycle.control_input_batch.zero(horizon=horizon),
            virtual=types.numpy.simple.control_input_batch.zero(
                horizon=horizon, dimension=1
            ),
        )

    @property
    def initial_state(self) -> MpccState:
        return types.numpy.augmented.state.of(
            physical=types.numpy.bicycle.state(x=0.0, y=0.0, heading=0.0, speed=0.0),
            virtual=types.numpy.simple.state(np.array([0.0])),
        )

    @property
    def nominal_input(self) -> MpccInputSequence:
        return types.numpy.augmented.control_input_sequence.of(
            physical=types.numpy.bicycle.control_input_sequence.zeroes(
                horizon=(horizon := 30)
            ),
            virtual=types.numpy.simple.control_input_sequence.zeroes(
                horizon=horizon, dimension=1
            ),
        )


class configure:
    @staticmethod
    def numpy_mpcc_planner_from_base(
        reference: Trajectory = REFERENCE,
    ) -> NumPyMpccPlannerConfiguration:
        # NOTE: Type Checkers like Pyright won't be able to infer complex types, so you may
        # need to help them with an explicit annotation.
        planner: Planner = mppi.numpy.base(
            model=(
                augmented_model := AugmentedModel.of(
                    physical=model.numpy.kinematic_bicycle(
                        time_step_size=(dt := 0.1),
                        wheelbase=(L := 2.5),
                        speed_limits=(0.0, 15.0),
                        steering_limits=(-0.5, 0.5),
                        acceleration_limits=(-3.0, 3.0),
                    ),
                    virtual=model.numpy.integrator(
                        time_step_size=dt,
                        state_limits=(0, reference.path_length),
                        velocity_limits=(0, 15),
                    ),
                    state=types.numpy.augmented.state,
                    batch=types.numpy.augmented.state_batch,
                )
            ),
            cost_function=costs.numpy.combined(
                contouring_cost := costs.numpy.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_extractor := extract.from_virtual(path_parameter)
                    ),
                    position_extractor=(
                        position_extractor := extract.from_physical(position)
                    ),
                    weight=50.0,
                ),
                costs.numpy.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=10.0,
                ),
                costs.numpy.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(path_velocity),
                    time_step_size=dt,
                    weight=250.0,
                ),
                costs.numpy.comfort.control_smoothing(
                    weights=np.array([2.0, 5.0, 5.0])
                ),
            ),
            sampler=AugmentedSampler.of(
                physical=sampler.numpy.gaussian(
                    standard_deviation=np.array([1.0, 1.0]),
                    rollout_count=(rollout_count := 512),
                    to_batch=types.numpy.bicycle.control_input_batch,
                    seed=42,
                ),
                virtual=sampler.numpy.gaussian(
                    standard_deviation=np.array([1.0]),
                    rollout_count=rollout_count,
                    to_batch=types.numpy.simple.control_input_batch,
                    seed=43,
                ),
                batch=types.numpy.augmented.control_input_batch,
            ),
        )

        return NumPyMpccPlannerConfiguration(
            reference=reference,
            planner=planner,
            model=augmented_model,
            contouring_cost=contouring_cost,
            wheelbase=L,
        )

    @staticmethod
    def numpy_mpcc_planner_from_augmented(
        reference: Trajectory = REFERENCE,
    ) -> NumPyMpccPlannerConfiguration:
        planner, augmented_model = mppi.numpy.augmented(
            models=(
                model.numpy.kinematic_bicycle(
                    time_step_size=(dt := 0.1),
                    wheelbase=(L := 2.5),
                    speed_limits=(0.0, 15.0),
                    steering_limits=(-0.5, 0.5),
                    acceleration_limits=(-3.0, 3.0),
                ),
                model.numpy.integrator(
                    time_step_size=dt,
                    state_limits=(0, reference.path_length),
                    velocity_limits=(0, 15),
                ),
            ),
            samplers=(
                sampler.numpy.gaussian(
                    standard_deviation=np.array([1.0, 1.0]),
                    rollout_count=(rollout_count := 512),
                    to_batch=types.numpy.bicycle.control_input_batch,
                    seed=42,
                ),
                sampler.numpy.gaussian(
                    standard_deviation=np.array([1.0]),
                    rollout_count=rollout_count,
                    to_batch=types.numpy.simple.control_input_batch,
                    seed=43,
                ),
            ),
            cost=costs.numpy.combined(
                contouring_cost := costs.numpy.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_extractor := extract.from_virtual(path_parameter)
                    ),
                    position_extractor=(
                        position_extractor := extract.from_physical(position)
                    ),
                    weight=50.0,
                ),
                costs.numpy.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=10.0,
                ),
                costs.numpy.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(path_velocity),
                    time_step_size=dt,
                    weight=250.0,
                ),
                costs.numpy.comfort.control_smoothing(
                    weights=np.array([2.0, 5.0, 5.0])
                ),
            ),
            state=types.numpy.augmented.state,
            state_batch=types.numpy.augmented.state_batch,
            input_batch=types.numpy.augmented.control_input_batch,
        )

        return NumPyMpccPlannerConfiguration(
            reference=reference,
            planner=planner,
            model=augmented_model,
            contouring_cost=contouring_cost,
            wheelbase=L,
        )
