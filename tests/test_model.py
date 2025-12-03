from trajax import bicycle, KinematicBicycleModel, NumPyBicycleModel, JaxBicycleModel

from numtypes import array, Array
import numpy as np

from tests.dsl import model as data, estimate, compute, clear_type
from pytest import mark


type ControlInputBatch = bicycle.ControlInputBatch
type State = bicycle.State
type StateBatch = bicycle.StateBatch


@mark.asyncio
@mark.parametrize(
    ["model", "inputs", "initial_state", "M", "T", "x_0", "y_0", "theta_0", "v_0"],
    [
        (
            model := NumPyBicycleModel.create(time_step_size=0.1),
            inputs := data.numpy.control_input_batch(
                time_horizon=(T := 5),
                rollout_count=(M := 3),
                acceleration=0.0,
                steering=0.0,
            ),
            initial_state := data.numpy.state(
                x=(x_0 := 15.0), y=(y_0 := 12.0), theta=(theta_0 := 0.5), v=(v_0 := 0.0)
            ),
            M,
            T,
            x_0,
            y_0,
            theta_0,
            v_0,
        ),
        (
            model := JaxBicycleModel.create(time_step_size=0.25),
            inputs := data.jax.control_input_batch(
                time_horizon=(T := 5),
                rollout_count=(M := 3),
                acceleration=0.0,
                steering=0.0,
            ),
            initial_state := data.jax.state(
                x=(x_0 := 20.0), y=(y_0 := 25.0), theta=(theta_0 := 0.1), v=(v_0 := 0.0)
            ),
            M,
            T,
            x_0,
            y_0,
            theta_0,
            v_0,
        ),
    ],
)
async def test_that_vehicle_position_does_not_change_when_velocity_and_input_are_zero[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    M: int,
    T: int,
    x_0: float,
    y_0: float,
    theta_0: float,
    v_0: float,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    assert np.allclose(
        rollouts,
        array([[[x_0] * M, [y_0] * M, [theta_0] * M, [v_0] * M]] * T, shape=(T, 4, M)),
    )
    assert np.allclose(
        positions := rollouts.positions,
        array([[[x_0] * M, [y_0] * M]] * T, shape=(T, 2, M)),
    )
    assert np.allclose(positions.x(), array([[x_0] * M] * T, shape=(T, M)))
    assert np.allclose(positions.y(), array([[y_0] * M] * T, shape=(T, M)))
    assert np.allclose(
        rollouts.orientations(), array([[theta_0] * M] * T, shape=(T, M))
    )
    assert np.allclose(rollouts.velocities(), array([[v_0] * M] * T, shape=(T, M)))


@mark.asyncio
@mark.parametrize(
    [
        "model",
        "inputs",
        "initial_state",
        "expected_x",
        "expected_y",
        "expected_theta",
        "expected_v",
    ],
    [
        (
            model := NumPyBicycleModel.create(time_step_size=(dt := 0.25)),
            inputs := data.numpy.control_input_batch(
                time_horizon=(T := 4),
                rollout_count=(M := 3),
                acceleration=0.0,
                steering=0.0,
            ),
            initial_state := data.numpy.state(
                x=(x_0 := 1.0),
                y=(y_0 := 2.0),
                # Choose theta, v, such that velocity components are (v_x, v_y)
                theta=(theta_0 := np.arctan2(v_y := 4, v_x := 2)),
                v=(v_0 := (v_x**2 + v_y**2) ** 0.5),
            ),
            expected_x := array(
                [
                    [x_0 + 1 * dt * v_x] * M,
                    [x_0 + 2 * dt * v_x] * M,
                    [x_0 + 3 * dt * v_x] * M,
                    [x_0 + 4 * dt * v_x] * M,
                ],
                shape=(T, M),
            ),
            expected_y := array(
                [
                    [y_0 + 1 * dt * v_y] * M,
                    [y_0 + 2 * dt * v_y] * M,
                    [y_0 + 3 * dt * v_y] * M,
                    [y_0 + 4 * dt * v_y] * M,
                ],
                shape=(T, M),
            ),
            expected_theta := array([[theta_0] * M] * T, shape=(T, M)),
            expected_v := array([[v_0] * M] * T, shape=(T, M)),
        ),
        (
            model := JaxBicycleModel.create(time_step_size=(dt := 0.75)),
            inputs := data.jax.control_input_batch(
                time_horizon=(T := 2),
                rollout_count=(M := 3),
                acceleration=0.0,
                steering=0.0,
            ),
            initial_state := data.jax.state(
                x=(x_0 := -5.0),
                y=(y_0 := 14.0),
                # Choose theta, v, such that velocity components are (v_x, v_y)
                theta=(theta_0 := np.arctan2(v_y := 4, v_x := 8)),
                v=(v_0 := (v_x**2 + v_y**2) ** 0.5),
            ),
            expected_x := array(
                [
                    [x_0 + 1 * dt * v_x] * M,
                    [x_0 + 2 * dt * v_x] * M,
                ],
                shape=(T, M),
            ),
            expected_y := array(
                [
                    [y_0 + 1 * dt * v_y] * M,
                    [y_0 + 2 * dt * v_y] * M,
                ],
                shape=(T, M),
            ),
            expected_theta := array([[theta_0] * M] * T, shape=(T, M)),
            expected_v := array([[v_0] * M] * T, shape=(T, M)),
        ),
    ],
)
async def test_that_vehicle_follows_straight_line_when_velocity_is_constant[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_x: Array,
    expected_y: Array,
    expected_theta: Array,
    expected_v: Array,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    assert np.allclose(rollouts.positions.x(), expected_x)
    assert np.allclose(rollouts.positions.y(), expected_y)
    assert np.allclose(rollouts.orientations(), expected_theta)
    assert np.allclose(rollouts.velocities(), expected_v)


@mark.asyncio
@mark.parametrize(
    [
        "model",
        "inputs",
        "initial_state",
        "expected_x_final",
        "expected_y_final",
        "expected_theta",
        "expected_v_middle",
        "expected_v_final",
    ],
    [  # Time step has to be small for 1st order integrators in these tests.
        (
            model := NumPyBicycleModel.create(time_step_size=(dt := 0.001)),
            inputs := data.numpy.control_input_batch(
                time_horizon=(T := 20),
                rollout_count=(M := 2),
                acceleration=(a := 1.0),
                steering=0.0,
            ),
            initial_state := data.numpy.state(
                x=(x_0 := 1.0),
                y=(y_0 := 2.0),
                theta=(theta_0 := np.pi / 6),
                v=(v_0 := 2.0),
            ),
            # Using the formula x = x_0 + v_x_0 * t + a_x * t^2 / 2
            expected_x_final := array(
                [
                    x_0
                    + (v_0 * np.cos(theta_0)) * (T * dt)
                    + (a * np.cos(theta_0)) * (T * dt) ** 2 / 2
                ]
                * M,
                shape=(M,),
            ),
            # Similarly for y
            expected_y_final := array(
                [
                    y_0
                    + (v_0 * np.sin(theta_0)) * (T * dt)
                    + (a * np.sin(theta_0)) * (T * dt) ** 2 / 2
                ]
                * M,
                shape=(M,),
            ),
            # Constant
            expected_theta := array([[theta_0] * M] * T, shape=(T, M)),
            # Using the formula v = v_0 + a * t
            expected_v_middle := array([v_0 + a * ((T // 2 + 1) * dt)] * M, shape=(M,)),
            expected_v_final := array([v_0 + a * (T * dt)] * M, shape=(M,)),
        ),
        (  # Similar test but using JAX implementation
            model := JaxBicycleModel.create(time_step_size=(dt := 0.001)),
            inputs := data.jax.control_input_batch(
                time_horizon=(T := 15),
                rollout_count=(M := 3),
                acceleration=(a := 2.0),
                steering=0.0,
            ),
            initial_state := data.jax.state(
                x=(x_0 := 4.0),
                y=(y_0 := -5.0),
                theta=(theta_0 := np.pi / 2),
                v=(v_0 := -2.0),
            ),
            expected_x_final := array(
                [
                    x_0
                    + (v_0 * np.cos(theta_0)) * (T * dt)
                    + (a * np.cos(theta_0)) * (T * dt) ** 2 / 2
                ]
                * M,
                shape=(M,),
            ),
            expected_y_final := array(
                [
                    y_0
                    + (v_0 * np.sin(theta_0)) * (T * dt)
                    + (a * np.sin(theta_0)) * (T * dt) ** 2 / 2
                ]
                * M,
                shape=(M,),
            ),
            expected_theta := array([[theta_0] * M] * T, shape=(T, M)),
            expected_v_middle := array([v_0 + a * ((T // 2 + 1) * dt)] * M, shape=(M,)),
            expected_v_final := array([v_0 + a * (T * dt)] * M, shape=(M,)),
        ),
    ],
)
async def test_that_vehicle_follows_straight_line_when_acceleration_is_constant[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_x_final: Array,
    expected_y_final: Array,
    expected_theta: Array,
    expected_v_middle: Array,
    expected_v_final: Array,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)
    T = inputs.horizon

    assert np.allclose(rollouts.positions.x()[-1], expected_x_final, atol=1e-6)
    assert np.allclose(rollouts.positions.y()[-1], expected_y_final, atol=1e-6)
    assert np.allclose(rollouts.orientations(), expected_theta, atol=1e-6)
    assert np.allclose(rollouts.velocities()[(T // 2)], expected_v_middle, atol=1e-6)
    assert np.allclose(rollouts.velocities()[-1], expected_v_final, atol=1e-6)


T = clear_type


@mark.asyncio
@mark.parametrize(
    ["model", "inputs", "initial_state", "expected_final_theta"],
    [
        (
            # Reverse steering halfway through. Final orientation should be the same as start.
            model := NumPyBicycleModel.create(time_step_size=0.5),
            inputs := data.numpy.control_input_batch(
                rollout_count=(M := 8),
                acceleration=array([0.0] * (T := 6), shape=(T,)),
                steering=array([0.2, 0.4, 1.0, -1.0, -0.4, -0.2], shape=(T,)),
            ),
            initial_state := data.numpy.state(
                x=2.0, y=4.0, theta=(theta_0 := 1.23), v=1.0
            ),
            expected_final_theta := array([theta_0] * M, shape=(M,)),
        ),
        (
            model := JaxBicycleModel.create(time_step_size=0.25),
            inputs := data.jax.control_input_batch(
                rollout_count=(M := 8),
                acceleration=array([0.0] * (T := 6), shape=(T,)),
                steering=array([0.3, -0.6, 2.0, -2.0, 0.6, -0.3], shape=(T,)),
            ),
            initial_state := data.jax.state(
                x=2.0, y=4.0, theta=(theta_0 := 2.42), v=1.0
            ),
            expected_final_theta := array([theta_0] * M, shape=(M,)),
        ),
    ],
)
async def test_that_vehicle_orientation_returns_to_start_when_steering_is_reversed[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_final_theta: Array,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    assert np.allclose(rollouts.orientations()[-1], expected_final_theta, atol=1e-6)


@mark.asyncio
@mark.parametrize(
    ["model", "inputs", "initial_state", "expected_final_v"],
    [
        (
            # Reverse acceleration halfway through. Final velocity should be the same as start.
            model := NumPyBicycleModel.create(time_step_size=0.5),
            inputs := data.numpy.control_input_batch(
                rollout_count=(M := 8),
                acceleration=array([2.0, 1.0, 0.5, -0.5, -1.0, -2.0], shape=(T := 6,)),
                steering=array([0.0] * T, shape=(T,)),
            ),
            initial_state := data.numpy.state(x=2.0, y=4.0, theta=1.23, v=(v_0 := 1.0)),
            expected_final_v := array([v_0] * M, shape=(M,)),
        ),
        (
            model := JaxBicycleModel.create(time_step_size=0.25),
            inputs := data.jax.control_input_batch(
                rollout_count=(M := 8),
                acceleration=array([3.0, 1.0, 0.5, -0.5, -1.0, -3.0], shape=(T := 6,)),
                steering=array([0.0] * T, shape=(T,)),
            ),
            initial_state := data.jax.state(x=3.0, y=4.0, theta=2.42, v=(v_0 := 1.0)),
            expected_final_v := array([v_0] * M, shape=(M,)),
        ),
    ],
)
async def test_that_vehicle_velocity_returns_to_start_when_acceleration_is_reversed[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_final_v: Array,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    assert np.allclose(rollouts.velocities()[-1], expected_final_v, atol=1e-6)


@mark.asyncio
@mark.parametrize(
    [
        "model",
        "inputs",
        "initial_state",
        "expected_final_x",
        "expected_final_y",
        "expected_final_theta",
    ],
    [
        (
            # Reverse acceleration halfway through. Final position and orientation should be the same as start.
            model := NumPyBicycleModel.create(time_step_size=0.5),
            inputs := data.numpy.control_input_batch(
                rollout_count=(M := 8),
                acceleration=array(
                    [2.0, 1.0, 0.5, -0.5, -1.0, -2.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                    shape=(T := 12,),
                ),
                steering=array([0.0] * T, shape=(T,)),
            ),
            initial_state := data.numpy.state(
                x=(x_0 := 2.0), y=(y_0 := 4.0), theta=(theta_0 := 1.23), v=0
            ),
            expected_final_x := array([x_0] * M, shape=(M,)),
            expected_final_y := array([y_0] * M, shape=(M,)),
            expected_final_theta := array([theta_0] * M, shape=(M,)),
        ),
        (
            model := JaxBicycleModel.create(time_step_size=0.25),
            inputs := data.jax.control_input_batch(
                rollout_count=(M := 8),
                acceleration=array(
                    [1.5, 1.0, 0.5, -0.5, -1.0, -1.5, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5],
                    shape=(T := 12,),
                ),
                steering=array([0.0] * T, shape=(T,)),
            ),
            initial_state := data.jax.state(
                x=(x_0 := 3.0), y=(y_0 := 5.0), theta=(theta_0 := 2.42), v=0
            ),
            expected_final_x := array([x_0] * M, shape=(M,)),
            expected_final_y := array([y_0] * M, shape=(M,)),
            expected_final_theta := array([theta_0] * M, shape=(M,)),
        ),
    ],
)
async def test_that_vehicle_returns_to_starting_position_when_initially_not_moving_and_acceleration_is_reversed[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_final_x: Array,
    expected_final_y: Array,
    expected_final_theta: Array,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    assert np.allclose(rollouts.positions.x()[-1], expected_final_x, atol=1e-6)
    assert np.allclose(rollouts.positions.y()[-1], expected_final_y, atol=1e-6)
    assert np.allclose(rollouts.orientations()[-1], expected_final_theta, atol=1e-6)


@mark.asyncio
@mark.parametrize(
    ["model", "inputs", "initial_state", "time_step_size"],
    [  # Time step and maximum steering angle must be small for 1st order integrators in these tests.
        (
            model := NumPyBicycleModel.create(time_step_size=(dt := 0.1)),
            inputs := data.numpy.control_input_batch(
                rollout_count=(M := 2),
                acceleration=np.random.uniform(-1.0, 1.0, size=(T := 12)),
                steering=np.random.uniform(-0.1, 0.1, size=(T,)),
            ),
            initial_state := data.numpy.state(x=4.0, y=5.0, theta=1.2, v=2.0),
            time_step_size := dt,
        ),
        (
            model := JaxBicycleModel.create(time_step_size=(dt := 0.05)),
            inputs := data.jax.control_input_batch(
                rollout_count=(M := 2),
                acceleration=np.random.uniform(-1.0, 1.0, size=(T := 10)),
                steering=np.random.uniform(-0.2, 0.2, size=(T,)),
            ),
            initial_state := data.jax.state(x=1.0, y=2.0, theta=np.pi / 2, v=2.0),
            time_step_size := dt,
        ),
    ],
)
async def test_that_displacement_is_consistent_with_velocity_state[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    time_step_size: float,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    # Prepend initial state to get full trajectory
    full_x = np.insert(rollouts.positions.x(), 0, initial_state.x, axis=0)
    full_y = np.insert(rollouts.positions.y(), 0, initial_state.y, axis=0)
    full_theta = np.insert(rollouts.orientations(), 0, initial_state.theta, axis=0)
    full_v = np.insert(rollouts.velocities(), 0, initial_state.v, axis=0)

    # Actual displacements
    delta_x = np.diff(full_x, axis=0)
    delta_y = np.diff(full_y, axis=0)

    expected = estimate.displacements(
        velocities=full_v, orientations=full_theta, time_step_size=time_step_size
    )

    # Allowing O(dt²) error to allow 1st order integrators to pass.
    assert np.allclose(delta_x, expected.delta_x, atol=time_step_size**2 * 2), (
        f"X displacement error: max {np.max(np.abs(delta_x - expected.delta_x))}"
    )
    assert np.allclose(delta_y, expected.delta_y, atol=time_step_size**2 * 2), (
        f"Y displacement error: max {np.max(np.abs(delta_y - expected.delta_y))}"
    )


@mark.asyncio
@mark.parametrize(
    [
        "model",
        "inputs",
        "initial_state",
        "expected_final_x",
        "expected_final_y",
        "expected_final_theta",
    ],
    [
        (
            # Full circle with bicycle model:
            # angular_velocity = v * tan(steering) / L
            # For 2π rotation: T * dt * angular_velocity = 2π
            # So steering = atan(2π / (T * dt * v / L))
            model := NumPyBicycleModel.create(
                time_step_size=(dt := 0.1), wheelbase=(L := 2.0)
            ),
            inputs := data.numpy.control_input_batch(
                rollout_count=(M := 2),
                acceleration=array([0.0] * (T := 100), shape=(T,)),
                steering=array(
                    [steering := np.arctan(2 * np.pi / (T * dt * (v := 1.2) / L))] * T,
                    shape=(T,),
                ),
            ),
            initial_state := data.numpy.state(
                x=(x_0 := 2.4), y=(y_0 := 3.6), theta=(theta_0 := 0.5), v=v
            ),
            expected_final_x := array([x_0] * M, shape=(M,)),
            expected_final_y := array([y_0] * M, shape=(M,)),
            expected_final_theta := array([theta_0] * M, shape=(M,)),
        ),
        (  # Analogous test with JAX implementation
            model := JaxBicycleModel.create(
                time_step_size=(dt := 0.05), wheelbase=(L := 1.5)
            ),
            inputs := data.jax.control_input_batch(
                rollout_count=(M := 2),
                acceleration=array([0.0] * (T := 200), shape=(T,)),
                steering=array(
                    [steering := np.arctan(2 * np.pi / (T * dt * (v := 1.4) / L))] * T,
                    shape=(T,),
                ),
            ),
            initial_state := data.jax.state(
                x=(x_0 := 1.0), y=(y_0 := 1.0), theta=(theta_0 := 0.25), v=v
            ),
            expected_final_x := array([x_0] * M, shape=(M,)),
            expected_final_y := array([y_0] * M, shape=(M,)),
            expected_final_theta := array([theta_0] * M, shape=(M,)),
        ),
    ],
)
async def test_that_vehicle_returns_to_start_when_completing_a_circle_with_constant_steering[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_final_x: Array,
    expected_final_y: Array,
    expected_final_theta: Array,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    final_x = rollouts.positions.x()[-1]
    final_y = rollouts.positions.y()[-1]
    final_theta = rollouts.orientations()[-1]

    assert np.allclose(final_x, expected_final_x, atol=0.5)
    assert np.allclose(final_y, expected_final_y, atol=0.5)
    assert np.all(compute.angular_distance(final_theta, expected_final_theta) < 0.1)


@mark.asyncio
@mark.parametrize(
    ["model", "inputs", "initial_state", "time_step_size", "expected_angular_velocity"],
    [
        (
            # Angular velocity = v * tan(steering) / wheelbase
            model := NumPyBicycleModel.create(
                time_step_size=(dt := 0.1), wheelbase=(L := 2.5)
            ),
            inputs := data.numpy.control_input_batch(
                time_horizon=(T := 10),
                rollout_count=(M := 2),
                acceleration=0.0,
                steering=(delta := 0.5),
            ),
            initial_state := data.numpy.state(x=2.0, y=4.0, theta=5.0, v=(v := 2.0)),
            dt,
            expected_angular_velocity := array([v * np.tan(delta) / L] * M, shape=(M,)),
        ),
        (  # Analogous test with JAX implementation
            model := JaxBicycleModel.create(
                time_step_size=(dt := 0.1), wheelbase=(L := 3.0)
            ),
            inputs := data.jax.control_input_batch(
                time_horizon=(T := 10),
                rollout_count=(M := 4),
                acceleration=0.0,
                steering=(delta := 0.3),
            ),
            initial_state := data.jax.state(x=2.0, y=4.0, theta=8.0, v=(v := 4.0)),
            dt,
            expected_angular_velocity := array([v * np.tan(delta) / L] * M, shape=(M,)),
        ),
    ],
)
async def test_that_angular_velocity_depends_on_wheelbase[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    time_step_size: float,
    expected_angular_velocity: Array,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    theta = rollouts.orientations()
    theta_with_initial = np.insert(theta, 0, initial_state.theta, axis=0)
    angular_velocities = np.diff(theta_with_initial, axis=0) / time_step_size

    assert np.allclose(angular_velocities, expected_angular_velocity, atol=1e-6)


@mark.asyncio
@mark.parametrize(
    ["model", "inputs", "initial_state", "max_speed", "min_speed"],
    [
        *[
            (
                model := NumPyBicycleModel.create(
                    time_step_size=1.0, speed_limits=(v_min := -5.0, v_max := 10.0)
                ),
                inputs := data.numpy.control_input_batch(
                    time_horizon=(T := 20),
                    rollout_count=(M := 3),
                    acceleration=a,  # Without speed limits, v would reach T * a
                    steering=0.4,
                ),
                initial_state := data.numpy.state(x=3.0, y=4.0, theta=0.2, v=2.0),
                v_max,
                v_min,
            )
            for a in [1.0, -1.0]
        ],
        *[
            (
                model := JaxBicycleModel.create(
                    time_step_size=1.0, speed_limits=(v_min := -3.0, v_max := 8.0)
                ),
                inputs := data.jax.control_input_batch(
                    time_horizon=(T := 15),
                    rollout_count=(M := 2),
                    acceleration=a,
                    steering=0.0,
                ),
                initial_state := data.jax.state(x=6.0, y=5.0, theta=0.4, v=2.0),
                v_max,
                v_min,
            )
            for a in [1.0, -2.0]
        ],
    ],
)
async def test_that_velocity_is_clamped_to_speed_limits[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    max_speed: float,
    min_speed: float,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    velocities = rollouts.velocities()

    assert np.all(velocities <= max_speed + 1e-6)
    assert np.all(velocities >= min_speed - 1e-6)


@mark.asyncio
@mark.parametrize(
    ["model", "inputs", "initial_state", "expected_theta_change"],
    [
        *[
            (
                # Steering input is larger than max_steering, but should be clipped
                # angular velocity = v * tan(clipped_steering) / L
                # Total theta_change = angular_velocity * dt * T
                model := NumPyBicycleModel.create(
                    time_step_size=(dt := 1.0),
                    wheelbase=(L := 1.0),
                    steering_limits=(delta_min := -0.2, delta_max := 0.3),
                ),
                inputs := data.numpy.control_input_batch(
                    time_horizon=(T := 5),
                    rollout_count=(M := 4),
                    acceleration=0.0,
                    steering=delta,
                ),
                initial_state := data.numpy.state(
                    x=2.0, y=4.0, theta=2.0, v=(v := 1.0)
                ),
                expected_theta_change := array(
                    [
                        (v * np.tan(delta_max if expected == "max" else delta_min) / L)
                        * (dt * T)
                    ]
                    * M,
                    shape=(M,),
                ),
            )
            for (delta, expected) in [(1.0, "max"), (-1.0, "min")]
        ],
        *[
            (
                model := JaxBicycleModel.create(
                    time_step_size=(dt := 0.5),
                    wheelbase=(L := 2.0),
                    steering_limits=(delta_min := -0.3, delta_max := 0.4),
                ),
                inputs := data.jax.control_input_batch(
                    time_horizon=(T := 4),
                    rollout_count=(M := 2),
                    acceleration=0.0,
                    steering=delta,
                ),
                initial_state := data.jax.state(x=2.0, y=4.0, theta=2.0, v=(v := 2.0)),
                expected_theta_change := array(
                    [
                        (v * np.tan(delta_max if expected == "max" else delta_min) / L)
                        * (dt * T)
                    ]
                    * M,
                    shape=(M,),
                ),
            )
            for (delta, expected) in [(1.5, "max"), (-0.5, "min")]
        ],
    ],
)
async def test_that_steering_input_is_clipped_to_max_steering[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_theta_change: Array,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    final_theta = rollouts.orientations()[-1]
    actual_theta_change = final_theta - initial_state.theta

    assert np.allclose(actual_theta_change, expected_theta_change, atol=1e-6)


@mark.asyncio
@mark.parametrize(
    ["model", "inputs", "initial_state", "expected_velocity"],
    [
        *[
            (
                # Acceleration input is larger than max_acceleration, but should be clipped
                # Total velocity change = clipped_acceleration * dt * T
                model := NumPyBicycleModel.create(
                    time_step_size=(dt := 1.0),
                    acceleration_limits=(a_min := -2.0, a_max := 3.0),
                ),
                inputs := data.numpy.control_input_batch(
                    time_horizon=(T := 5),
                    rollout_count=(M := 2),
                    acceleration=a,
                    steering=0.0,
                ),
                initial_state := data.numpy.state(
                    x=0.0, y=0.0, theta=0.0, v=(v_0 := 0.0)
                ),
                expected_velocity := array(
                    [v_0 + (a_max if expected == "max" else a_min) * dt * T] * M,
                    shape=(M,),
                ),
            )
            for (a, expected) in [(10.0, "max"), (-5.0, "min")]
        ],
        *[
            (
                model := JaxBicycleModel.create(
                    time_step_size=(dt := 0.5),
                    acceleration_limits=(a_min := -8.0, a_max := 4.0),
                ),
                inputs := data.jax.control_input_batch(
                    time_horizon=(T := 5),
                    rollout_count=(M := 2),
                    acceleration=a,
                    steering=0.0,
                ),
                initial_state := data.jax.state(
                    x=0.0, y=0.0, theta=0.0, v=(v_0 := 1.0)
                ),
                expected_velocity := array(
                    [v_0 + (a_max if expected == "max" else a_min) * dt * T] * M,
                    shape=(M,),
                ),
            )
            for (a, expected) in [(5.0, "max"), (-10.0, "min")]
        ],
    ],
)
async def test_that_acceleration_input_is_clipped_to_max_acceleration[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](
    model: KinematicBicycleModel[ControlInputBatchT, StateT, StateBatchT],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_velocity: Array,
) -> None:
    rollouts = await model.simulate(inputs, initial_state)

    final_velocity = rollouts.velocities()[-1]

    assert np.allclose(final_velocity, expected_velocity, atol=1e-6)
