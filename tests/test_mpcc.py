from typing import cast

from trajax import (
    AugmentedModel,
    AugmentedSampler,
    Trajectory,
    NumPyMppi,
    mppi,
    model,
    sampler,
    costs,
    trajectory,
    types,
)

import numpy as np
from numtypes import array, Array, Dim2

from pytest import mark


type MpccPhysicalState = types.numpy.bicycle.State
type MpccVirtualState = types.numpy.simple.State
type MpccState = types.numpy.augmented.State[MpccPhysicalState, MpccVirtualState]

type MpccPhysicalStateBatch = types.numpy.bicycle.StateBatch
type MpccVirtualStateBatch = types.numpy.simple.StateBatch
type MpccStateBatch = types.numpy.augmented.StateBatch[
    MpccPhysicalStateBatch, MpccVirtualStateBatch
]

type MpccPhysicalInputSequence = types.numpy.bicycle.ControlInputSequence
type MpccVirtualInputSequence = types.numpy.simple.ControlInputSequence
type MpccInputSequence = types.numpy.augmented.ControlInputSequence[
    MpccPhysicalInputSequence, MpccVirtualInputSequence
]

type MpccPhysicalInputBatch = types.numpy.bicycle.ControlInputBatch
type MpccVirtualInputBatch = types.numpy.simple.ControlInputBatch
type MpccInputBatch = types.numpy.augmented.ControlInputBatch[
    MpccPhysicalInputBatch, MpccVirtualInputBatch
]
type MpccCosts = types.numpy.Costs

type MpccDynamicalModel = AugmentedModel[
    MpccPhysicalState,
    MpccPhysicalState,
    MpccPhysicalStateBatch,
    MpccPhysicalInputSequence,
    MpccPhysicalInputBatch,
    MpccVirtualState,
    MpccVirtualState,
    MpccVirtualStateBatch,
    MpccVirtualInputSequence,
    MpccVirtualInputBatch,
]
type MpccMppi = NumPyMppi[
    MpccState,
    MpccState,
    MpccStateBatch,
    MpccInputSequence,
    MpccInputSequence,
    MpccInputBatch,
    MpccCosts,
]


def compute_lateral_deviation(
    position: tuple[float, float],
    ref_trajectory: Trajectory,
    path_parameter: float,
) -> float:
    """Computes the lateral deviation from the reference trajectory."""
    ref_point = ref_trajectory.query(
        types.numpy.path_parameters(array([[path_parameter]], shape=(1, 1)))
    )
    ref_x = float(ref_point.x()[0, 0])
    ref_y = float(ref_point.y()[0, 0])
    heading = float(ref_point.heading()[0, 0])

    dx = position[0] - ref_x
    dy = position[1] - ref_y

    # Lateral error is perpendicular to the path direction
    lateral_error = abs(np.sin(heading) * dx - np.cos(heading) * dy)
    return lateral_error


@mark.asyncio
async def test_that_mpcc_planner_follows_trajectory_without_excessive_deviation() -> (
    None
):
    planner = cast(MpccMppi, mppi.numpy())
    physical_model = model.numpy.kinematic_bicycle(
        time_step_size=(dt := 0.1),
        wheelbase=2.5,
        speed_limits=(0.0, 15.0),
        steering_limits=(-0.5, 0.5),
        acceleration_limits=(-3.0, 3.0),
    )
    virtual_model = model.numpy.integrator(
        time_step_size=dt, state_limits=(0, path_length := 60), velocity_limits=(0, 15)
    )
    augmented_model: MpccDynamicalModel = AugmentedModel.of(
        physical=physical_model,
        virtual=virtual_model,
        control_dimension=3,
        state_dimension=5,
    )

    augmented_sampler = AugmentedSampler.of(
        physical=sampler.numpy(
            standard_deviation=np.array([1.0, 1.0]),
            rollout_count=(rollout_count := 512),
            to_batch=types.numpy.bicycle.control_input_batch,
            seed=42,
        ),
        virtual=sampler.numpy(
            standard_deviation=np.array([1.0]),
            rollout_count=rollout_count,
            to_batch=types.numpy.simple.control_input_batch,
            seed=43,
        ),
        rollout_count=rollout_count,
    )
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
        path_length=path_length,
    )

    def path_parameter(states: MpccVirtualStateBatch) -> types.numpy.PathParameters:
        return types.numpy.path_parameters(states.array[:, 0, :])

    def path_velocity(inputs: MpccVirtualInputBatch) -> Array[Dim2]:
        return inputs.array[:, 0, :]

    def position(states: MpccPhysicalStateBatch) -> types.numpy.Positions:
        return types.numpy.positions(x=states.positions.x(), y=states.positions.y())

    path_parameter_extractor = types.augmented.state_batch.from_virtual(path_parameter)
    path_velocity_extractor = types.augmented.control_input_batch.from_virtual(
        path_velocity
    )
    position_extractor = types.augmented.state_batch.from_physical(position)

    combined_cost = costs.numpy.combined(
        costs.numpy.tracking.contouring(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=50.0,
        ),
        costs.numpy.tracking.lag(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=10.0,
        ),
        costs.numpy.tracking.progress(
            path_velocity_extractor=path_velocity_extractor,
            time_step_size=dt,
            weight=5.0,
        ),
        costs.numpy.comfort.control_smoothing(weights=np.array([0.1, 0.1, 0.01])),
    )

    current_state = types.augmented.state.of(
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

    min_progress = path_length * 0.75
    progress = 0.0

    for step in range(300):
        control = await planner.step(
            model=augmented_model,
            cost_function=combined_cost,
            sampler=augmented_sampler,
            temperature=0.05,
            nominal_input=nominal_input,
            initial_state=current_state,
        )

        nominal_input = control.nominal

        current_state = await augmented_model.step(
            input=control.optimal, state=current_state
        )

        lateral_deviation = compute_lateral_deviation(
            position=(current_state.physical.x, current_state.physical.y),
            ref_trajectory=reference,
            path_parameter=current_state.virtual.array[0],
        )

        assert lateral_deviation <= 1.5, (
            f"Lateral deviation exceeded at step {step}: {lateral_deviation} m"
        )

        if (progress := current_state.virtual.array[0]) >= min_progress:
            break

    assert progress > min_progress, (
        f"Vehicle did not make sufficient progress along the path. "
        f"Final path parameter: {progress:.1f}, expected > {min_progress:.1f}"
    )
