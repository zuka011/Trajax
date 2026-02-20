from typing import Callable, Sequence

from faran import types, DynamicalModel, model as create_model

from numtypes import array, Array

import numpy as np

from tests.dsl import model as data, estimate, compute
from pytest import mark


type State = types.bicycle.State
type StateSequence = types.bicycle.StateSequence
type StateBatch = types.bicycle.StateBatch
type ControlInputSequence = types.bicycle.ControlInputSequence
type ControlInputBatch = types.bicycle.ControlInputBatch


class test_that_vehicle_position_does_not_change_when_velocity_and_input_are_zero:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.bicycle.dynamical(time_step_size=0.1),
                inputs := data.control_input_batch(
                    time_horizon=(T := 5),
                    rollout_count=(M := 3),
                    acceleration=0.0,
                    steering=0.0,
                ),
                initial_state := data.state(
                    x=(x_0 := 15.0),
                    y=(y_0 := 12.0),
                    heading=(theta_0 := 0.5),
                    speed=(v_0 := 0.0),
                ),
                M,
                T,
                x_0,
                y_0,
                theta_0,
                v_0,
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "M", "T", "x_0", "y_0", "theta_0", "v_0"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        M: int,
        T: int,
        x_0: float,
        y_0: float,
        theta_0: float,
        v_0: float,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(
            rollouts,
            array(
                [[[x_0] * M, [y_0] * M, [theta_0] * M, [v_0] * M]] * T, shape=(T, 4, M)
            ),
        )
        assert np.allclose(
            positions := rollouts.positions,
            array([[[x_0] * M, [y_0] * M]] * T, shape=(T, 2, M)),
        )
        assert np.allclose(positions.x(), array([[x_0] * M] * T, shape=(T, M)))
        assert np.allclose(positions.y(), array([[y_0] * M] * T, shape=(T, M)))
        assert np.allclose(rollouts.heading(), array([[theta_0] * M] * T, shape=(T, M)))
        assert np.allclose(rollouts.speed(), array([[v_0] * M] * T, shape=(T, M)))


class test_that_vehicle_follows_straight_line_when_velocity_is_constant:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.bicycle.dynamical(time_step_size=(dt := 0.25)),
                inputs := data.control_input_batch(
                    time_horizon=(T := 4),
                    rollout_count=(M := 3),
                    acceleration=0.0,
                    steering=0.0,
                ),
                initial_state := data.state(
                    x=(x_0 := 1.0),
                    y=(y_0 := 2.0),
                    # Choose theta, v, such that velocity components are (v_x, v_y)
                    heading=(theta_0 := np.arctan2(v_y := 4, v_x := 2)),
                    speed=(v_0 := (v_x**2 + v_y**2) ** 0.5),
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
        ]

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
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        expected_x: Array,
        expected_y: Array,
        expected_theta: Array,
        expected_v: Array,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts.positions.x(), expected_x)
        assert np.allclose(rollouts.positions.y(), expected_y)
        assert np.allclose(rollouts.heading(), expected_theta)
        assert np.allclose(rollouts.speed(), expected_v)


class test_that_vehicle_follows_straight_line_when_acceleration_is_constant:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (  # Time step has to be small for 1st order integrators in these tests.
                model := create_model.bicycle.dynamical(time_step_size=(dt := 0.001)),
                inputs := data.control_input_batch(
                    time_horizon=(T := 20),
                    rollout_count=(M := 2),
                    acceleration=(a := 1.0),
                    steering=0.0,
                ),
                initial_state := data.state(
                    x=(x_0 := 1.0),
                    y=(y_0 := 2.0),
                    heading=(theta_0 := np.pi / 6),
                    speed=(v_0 := 2.0),
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
                expected_v_middle := array(
                    [v_0 + a * ((T // 2 + 1) * dt)] * M, shape=(M,)
                ),
                expected_v_final := array([v_0 + a * (T * dt)] * M, shape=(M,)),
            ),
        ]

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
        [
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT: ControlInputBatch,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        expected_x_final: Array,
        expected_y_final: Array,
        expected_theta: Array,
        expected_v_middle: Array,
        expected_v_final: Array,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)
        T = inputs.horizon

        assert np.allclose(rollouts.positions.x()[-1], expected_x_final, atol=1e-6)
        assert np.allclose(rollouts.positions.y()[-1], expected_y_final, atol=1e-6)
        assert np.allclose(rollouts.heading(), expected_theta, atol=1e-6)
        assert np.allclose(rollouts.speed()[(T // 2)], expected_v_middle, atol=1e-6)
        assert np.allclose(rollouts.speed()[-1], expected_v_final, atol=1e-6)


class test_that_vehicle_orientation_returns_to_start_when_steering_is_reversed:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (  # Reverse steering halfway through. Final orientation should be the same as start.
                model := create_model.bicycle.dynamical(time_step_size=0.5),
                inputs := data.control_input_batch(
                    rollout_count=(M := 8),
                    acceleration=array([0.0] * (T := 6), shape=(T,)),
                    steering=array([0.2, 0.4, 1.0, -1.0, -0.4, -0.2], shape=(T,)),
                ),
                initial_state := data.state(
                    x=2.0, y=4.0, heading=(theta_0 := 1.23), speed=1.0
                ),
                expected_final_theta := array([theta_0] * M, shape=(M,)),
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_final_theta"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        expected_final_theta: Array,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts.heading()[-1], expected_final_theta, atol=1e-6)


class test_that_vehicle_velocity_returns_to_start_when_acceleration_is_reversed:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (  # Reverse acceleration halfway through. Final velocity should be the same as start.
                model := create_model.bicycle.dynamical(time_step_size=0.5),
                inputs := data.control_input_batch(
                    rollout_count=(M := 8),
                    acceleration=array(
                        [2.0, 1.0, 0.5, -0.5, -1.0, -2.0], shape=(T := 6,)
                    ),
                    steering=array([0.0] * T, shape=(T,)),
                ),
                initial_state := data.state(
                    x=2.0, y=4.0, heading=1.23, speed=(v_0 := 1.0)
                ),
                expected_final_v := array([v_0] * M, shape=(M,)),
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_final_v"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        expected_final_v: Array,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts.speed()[-1], expected_final_v, atol=1e-6)


class test_that_vehicle_returns_to_starting_position_when_initially_not_moving_and_acceleration_is_reversed:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                # Reverse acceleration halfway through. Final position and orientation should be the same as start.
                model := create_model.bicycle.dynamical(time_step_size=0.5),
                inputs := data.control_input_batch(
                    rollout_count=(M := 8),
                    acceleration=array(
                        [2.0, 1.0, 0.5, -0.5, -1.0, -2.0]
                        + [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0],
                        shape=(T := 12,),
                    ),
                    steering=array([0.0] * T, shape=(T,)),
                ),
                initial_state := data.state(
                    x=(x_0 := 2.0), y=(y_0 := 4.0), heading=(theta_0 := 1.23), speed=0
                ),
                expected_final_x := array([x_0] * M, shape=(M,)),
                expected_final_y := array([y_0] * M, shape=(M,)),
                expected_final_theta := array([theta_0] * M, shape=(M,)),
            ),
        ]

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
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        expected_final_x: Array,
        expected_final_y: Array,
        expected_final_theta: Array,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts.positions.x()[-1], expected_final_x, atol=1e-6)
        assert np.allclose(rollouts.positions.y()[-1], expected_final_y, atol=1e-6)
        assert np.allclose(rollouts.heading()[-1], expected_final_theta, atol=1e-6)


class test_that_displacement_is_consistent_with_velocity_state:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (  # Time step and maximum steering angle must be small for 1st order integrators in these tests.
                model := create_model.bicycle.dynamical(time_step_size=(dt := 0.1)),
                inputs := data.control_input_batch(
                    rollout_count=(M := 2),
                    acceleration=np.random.uniform(-1.0, 1.0, size=(T := 12)),
                    steering=np.random.uniform(-0.1, 0.1, size=(T,)),
                ),
                initial_state := data.state(x=4.0, y=5.0, heading=1.2, speed=2.0),
                time_step_size := dt,
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "time_step_size"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        time_step_size: float,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        # Prepend initial state to get full trajectory
        full_x = np.insert(rollouts.positions.x(), 0, initial_state.x, axis=0)
        full_y = np.insert(rollouts.positions.y(), 0, initial_state.y, axis=0)
        full_theta = np.insert(rollouts.heading(), 0, initial_state.heading, axis=0)
        full_v = np.insert(rollouts.speed(), 0, initial_state.speed, axis=0)

        # Actual displacements
        delta_x = np.diff(full_x, axis=0)
        delta_y = np.diff(full_y, axis=0)

        expected = estimate.displacements(
            velocities=full_v, heading=full_theta, time_step_size=time_step_size
        )

        # Allowing O(dt²) error to allow 1st order integrators to pass.
        assert np.allclose(delta_x, expected.delta_x, atol=time_step_size**2 * 2), (
            f"X displacement error: max {np.max(np.abs(delta_x - expected.delta_x))}"
        )
        assert np.allclose(delta_y, expected.delta_y, atol=time_step_size**2 * 2), (
            f"Y displacement error: max {np.max(np.abs(delta_y - expected.delta_y))}"
        )


class test_that_vehicle_returns_to_start_when_completing_a_circle_with_constant_steering:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                # Full circle with bicycle model:
                # angular_velocity = v * tan(steering) / L
                # For 2π rotation: T * dt * angular_velocity = 2π
                # So steering = atan(2π / (T * dt * v / L))
                model := create_model.bicycle.dynamical(
                    time_step_size=(dt := 0.1), wheelbase=(L := 2.0)
                ),
                inputs := data.control_input_batch(
                    rollout_count=(M := 2),
                    acceleration=array([0.0] * (T := 100), shape=(T,)),
                    steering=array(
                        [steering := np.arctan(2 * np.pi / (T * dt * (v := 1.2) / L))]
                        * T,
                        shape=(T,),
                    ),
                ),
                initial_state := data.state(
                    x=(x_0 := 2.4), y=(y_0 := 3.6), heading=(theta_0 := 0.5), speed=v
                ),
                expected_final_x := array([x_0] * M, shape=(M,)),
                expected_final_y := array([y_0] * M, shape=(M,)),
                expected_final_theta := array([theta_0] * M, shape=(M,)),
            ),
        ]

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
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        expected_final_x: Array,
        expected_final_y: Array,
        expected_final_theta: Array,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        final_x = rollouts.positions.x()[-1]
        final_y = rollouts.positions.y()[-1]
        final_theta = rollouts.heading()[-1]

        assert np.allclose(final_x, expected_final_x, atol=0.5)
        assert np.allclose(final_y, expected_final_y, atol=0.5)
        assert np.all(compute.angular_distance(final_theta, expected_final_theta) < 0.1)


class test_that_angular_velocity_depends_on_wheelbase:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (  # Angular velocity = v * tan(steering) / wheelbase
                model := create_model.bicycle.dynamical(
                    time_step_size=(dt := 0.1), wheelbase=(L := 2.5)
                ),
                inputs := data.control_input_batch(
                    time_horizon=(T := 10),
                    rollout_count=(M := 2),
                    acceleration=0.0,
                    steering=(delta := 0.5),
                ),
                initial_state := data.state(
                    x=2.0, y=4.0, heading=5.0, speed=(v := 2.0)
                ),
                dt,
                expected_angular_velocity := array(
                    [v * np.tan(delta) / L] * M, shape=(M,)
                ),
            ),
        ]

    @mark.parametrize(
        [
            "model",
            "inputs",
            "initial_state",
            "time_step_size",
            "expected_angular_velocity",
        ],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        time_step_size: float,
        expected_angular_velocity: Array,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        theta = rollouts.heading()
        theta_with_initial = np.insert(theta, 0, initial_state.heading, axis=0)
        angular_velocities = np.diff(theta_with_initial, axis=0) / time_step_size

        assert np.allclose(angular_velocities, expected_angular_velocity, atol=1e-6)


class test_that_velocity_is_clamped_to_speed_limits:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            *[
                (
                    model := create_model.bicycle.dynamical(
                        time_step_size=1.0, speed_limits=(v_min := -5.0, v_max := 10.0)
                    ),
                    inputs := data.control_input_batch(
                        time_horizon=(T := 20),
                        rollout_count=(M := 3),
                        acceleration=a,  # Without speed limits, v would reach T * a
                        steering=0.4,
                    ),
                    initial_state := data.state(x=3.0, y=4.0, heading=0.2, speed=2.0),
                    v_max,
                    v_min,
                )
                for a in [1.0, -1.0]
            ],
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "max_speed", "min_speed"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        max_speed: float,
        min_speed: float,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        speed = rollouts.speed()

        assert np.all(speed <= max_speed + 1e-6)
        assert np.all(speed >= min_speed - 1e-6)


class test_that_steering_input_is_clipped_to_max_steering:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            *[
                (
                    # Steering input is larger than max_steering, but should be clipped
                    # angular velocity = v * tan(clipped_steering) / L
                    # Total theta_change = angular_velocity * dt * T
                    model := create_model.bicycle.dynamical(
                        time_step_size=(dt := 1.0),
                        wheelbase=(L := 1.0),
                        steering_limits=(delta_min := -0.2, delta_max := 0.3),
                    ),
                    inputs := data.control_input_batch(
                        time_horizon=(T := 5),
                        rollout_count=(M := 4),
                        acceleration=0.0,
                        steering=delta,
                    ),
                    initial_state := data.state(
                        x=2.0, y=4.0, heading=2.0, speed=(v := 1.0)
                    ),
                    expected_theta_change := array(
                        [
                            (
                                v
                                * np.tan(delta_max if expected == "max" else delta_min)
                                / L
                            )
                            * (dt * T)
                        ]
                        * M,
                        shape=(M,),
                    ),
                )
                for (delta, expected) in [(1.0, "max"), (-1.0, "min")]
            ],
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_theta_change"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT: State,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        expected_theta_change: Array,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        final_theta = rollouts.heading()[-1]
        actual_theta_change = final_theta - initial_state.heading

        assert np.allclose(actual_theta_change, expected_theta_change, atol=1e-6)


class test_that_acceleration_input_is_clipped_to_max_acceleration:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            *[
                (
                    # Acceleration input is larger than max_acceleration, but should be clipped
                    # Total velocity change = clipped_acceleration * dt * T
                    model := create_model.bicycle.dynamical(
                        time_step_size=(dt := 1.0),
                        acceleration_limits=(a_min := -2.0, a_max := 3.0),
                    ),
                    inputs := data.control_input_batch(
                        time_horizon=(T := 5),
                        rollout_count=(M := 2),
                        acceleration=a,
                        steering=0.0,
                    ),
                    initial_state := data.state(
                        x=0.0, y=0.0, heading=0.0, speed=(v_0 := 0.0)
                    ),
                    expected_velocity := array(
                        [v_0 + (a_max if expected == "max" else a_min) * dt * T] * M,
                        shape=(M,),
                    ),
                )
                for (a, expected) in [(10.0, "max"), (-5.0, "min")]
            ],
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_velocity"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.bicycle),
            *cases(create_model=create_model.jax, data=data.jax.bicycle),
        ],
    )
    def test[
        StateT,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        inputs: ControlInputBatchT,
        initial_state: StateT,
        expected_velocity: Array,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        final_velocity = rollouts.speed()[-1]

        assert np.allclose(final_velocity, expected_velocity, atol=1e-6)


class test_that_simulating_individual_steps_matches_horizon_simulation:
    @staticmethod
    def cases(create_model, data, types) -> Sequence[tuple]:
        return [
            (
                model := create_model.bicycle.dynamical(
                    time_step_size=(dt := 0.1),
                    wheelbase=1.5,
                    speed_limits=(0.0, 20.0),
                    steering_limits=(-0.4, 0.4),
                    acceleration_limits=(-5.0, 5.0),
                ),
                input_batch := data.control_input_batch(
                    rollout_count=1,
                    acceleration=array([2.0, 1.5, -0.5, 0.0, 1.0], shape=(T := 5,)),
                    steering=array([0.1, -0.1, 0.2, 0.0, -0.2], shape=(T,)),
                ),
                initial_state := data.state(x=0.0, y=0.0, heading=0.0, speed=5.0),
                horizon := T,
                input_of := lambda input_batch, t: types.bicycle.control_input_sequence(
                    input_batch.array[t:, :, 0]
                ),
            ),
        ]

    @mark.parametrize(
        ["model", "input_batch", "initial_state", "horizon", "input_of"],
        [
            *cases(
                create_model=create_model.numpy,
                data=data.numpy.bicycle,
                types=types.numpy,
            ),
            *cases(
                create_model=create_model.jax, data=data.jax.bicycle, types=types.jax
            ),
        ],
    )
    def test[
        StateT: State,
        StateSequenceT,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        input_batch: ControlInputBatchT,
        initial_state: StateT,
        horizon: int,
        input_of: Callable[[ControlInputBatchT, int], ControlInputSequenceT],
    ) -> None:
        rollout = (model.simulate(input_batch, initial_state)).rollout(0)
        current_state = initial_state

        for t in range(horizon):
            current_state = model.step(input_of(input_batch, t), current_state)

            assert np.allclose(current_state, rollout.step(t), atol=1e-6), (
                f"Mismatch at time step {t}: expected {rollout.step(t)}, got {current_state}"
            )


class test_that_simulating_individual_input_sequence_matches_horizon_simulation:
    @staticmethod
    def cases(create_model, data, types) -> Sequence[tuple]:
        return [
            (
                model := create_model.bicycle.dynamical(
                    time_step_size=(dt := 0.1),
                    wheelbase=1.5,
                    speed_limits=(0.0, 20.0),
                    steering_limits=(-0.4, 0.4),
                    acceleration_limits=(-5.0, 5.0),
                ),
                input_batch := data.control_input_batch(
                    rollout_count=1,
                    acceleration=array([1.0, 0.5, -0.5, 0.0], shape=(T := 4,)),
                    steering=array([0.1, -0.1, 0.2, 0.0], shape=(T,)),
                ),
                initial_state := data.state(x=0.0, y=0.0, heading=0.0, speed=2.0),
                input_of := lambda input_batch, t: types.bicycle.control_input_sequence(
                    input_batch.array[..., t]
                ),
            ),
        ]

    @mark.parametrize(
        ["model", "input_batch", "initial_state", "input_of"],
        [
            *cases(
                create_model=create_model.numpy,
                data=data.numpy.bicycle,
                types=types.numpy,
            ),
            *cases(
                create_model=create_model.jax, data=data.jax.bicycle, types=types.jax
            ),
        ],
    )
    def test[
        StateT,
        StateSequenceT: StateSequence,
        StateBatchT: StateBatch,
        ControlInputSequenceT,
        ControlInputBatchT,
    ](
        self,
        model: DynamicalModel[
            StateT,
            StateSequenceT,
            StateBatchT,
            ControlInputSequenceT,
            ControlInputBatchT,
        ],
        input_batch: ControlInputBatchT,
        initial_state: StateT,
        input_of: Callable[[ControlInputBatchT, int], ControlInputSequenceT],
    ) -> None:
        rollout = (model.simulate(input_batch, initial_state)).rollout(m := 0)
        sequence = model.forward(input_of(input_batch, m), initial_state)

        assert np.allclose(sequence, rollout, atol=1e-6)
