from trajax import (
    NumPyMppi,
    AugmentedModel,
    AugmentedSampler,
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

from pytest import mark

type PhysicalState = types.numpy.bicycle.State
type PhysicalStateBatch = types.numpy.bicycle.StateBatch
type PhysicalInputSequence = types.numpy.bicycle.ControlInputSequence
type PhysicalInputBatch = types.numpy.bicycle.ControlInputBatch
type VirtualState = types.numpy.simple.State
type VirtualStateBatch = types.numpy.simple.StateBatch
type VirtualInputSequence = types.numpy.simple.ControlInputSequence
type VirtualInputBatch = types.numpy.simple.ControlInputBatch
type State = types.numpy.augmented.State[PhysicalState, VirtualState]
type StateBatch = types.numpy.augmented.StateBatch[
    PhysicalStateBatch, VirtualStateBatch
]
type InputSequence = types.numpy.augmented.ControlInputSequence[
    PhysicalInputSequence, VirtualInputSequence
]
type InputBatch = types.numpy.augmented.ControlInputBatch[
    PhysicalInputBatch, VirtualInputBatch
]
type CombinedCost = types.CostFunction[InputBatch, StateBatch, types.numpy.Costs]
type Planner = NumPyMppi[InputSequence, InputSequence]


@mark.asyncio
async def test_that_mpcc_planner_follows_trajectory_without_excessive_deviation() -> (
    None
):
    reference = trajectory.numpy.waypoints(
        points=array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [20.0, 5.0],
                [30.0, 5.0],
                [40.0, 0.0],
                [50.0, 0.0],
            ],
            shape=(6, 2),
        ),
        path_length=(path_length := 60.0),
    )

    planner: Planner = mppi.numpy.base()
    augmented_model = AugmentedModel.of(
        physical=model.numpy.kinematic_bicycle(
            time_step_size=(dt := 0.1),
            wheelbase=2.5,
            speed_limits=(0.0, 15.0),
            steering_limits=(-0.5, 0.5),
            acceleration_limits=(-3.0, 3.0),
        ),
        virtual=model.numpy.integrator(
            time_step_size=dt,
            state_limits=(0, path_length),
            velocity_limits=(0, 15),
        ),
        state=types.numpy.augmented.state,
        batch=types.numpy.augmented.state_batch,
    )

    augmented_sampler = AugmentedSampler.of(
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
    )

    def path_parameter(states: VirtualStateBatch) -> types.numpy.PathParameters:
        return types.numpy.path_parameters(states.array[:, 0, :])

    def path_velocity(inputs: VirtualInputBatch) -> Array[Dim2]:
        return inputs.array[:, 0, :]

    def position(states: PhysicalStateBatch) -> types.numpy.Positions:
        return types.numpy.positions(x=states.positions.x(), y=states.positions.y())

    combined_cost: CombinedCost = costs.numpy.combined(
        contouring_cost := costs.numpy.tracking.contouring(
            reference=reference,
            path_parameter_extractor=(
                path_extractor := extract.from_virtual(path_parameter)
            ),
            position_extractor=(position_extractor := extract.from_physical(position)),
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
            weight=5.0,
        ),
        costs.numpy.comfort.control_smoothing(weights=np.array([0.1, 0.1, 0.01])),
    )

    current_state = types.numpy.augmented.state.of(
        physical=types.numpy.bicycle.state(x=0.0, y=0.0, heading=0.0, speed=5.0),
        virtual=types.numpy.simple.state(np.array([0.0])),
    )

    nominal_input = types.numpy.augmented.control_input_sequence.of(
        physical=types.numpy.bicycle.control_input_sequence.zeroes(
            horizon=(horizon := 30)
        ),
        virtual=types.numpy.simple.control_input_sequence.zeroes(
            horizon=horizon, dimension=1
        ),
    )

    states: list[State] = []
    min_progress = path_length * 0.75
    progress = 0.0

    for _ in range(300):
        control = await planner.step(
            model=augmented_model,
            cost_function=combined_cost,
            sampler=augmented_sampler,
            temperature=0.05,
            nominal_input=nominal_input,
            initial_state=current_state,
        )

        nominal_input = control.nominal

        states.append(
            current_state := await augmented_model.step(
                input=control.optimal, state=current_state
            )
        )

        if (progress := current_state.virtual.array[0]) >= min_progress:
            break

    assert progress > min_progress, (
        f"Vehicle did not make sufficient progress along the path. "
        f"Final path parameter: {progress:.1f}, expected > {min_progress:.1f}"
    )

    # TODO: Add Lateral Deviation Check.
