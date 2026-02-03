from typing import Callable, Sequence

from trajax import types, DynamicalModel, model as create_model

from numtypes import array, Array

import numpy as np

from tests.dsl import model as data, compute
from pytest import mark


type State = types.unicycle.State
type StateSequence = types.unicycle.StateSequence
type StateBatch = types.unicycle.StateBatch
type ControlInputSequence = types.unicycle.ControlInputSequence
type ControlInputBatch = types.unicycle.ControlInputBatch


class test_that_vehicle_position_does_not_change_when_velocity_is_zero:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.unicycle.dynamical(time_step_size=0.1),
                inputs := data.control_input_batch(
                    time_horizon=(T := 5),
                    rollout_count=(M := 3),
                    linear_velocity=0.0,
                    angular_velocity=0.0,
                ),
                initial_state := data.state(
                    x=(x_0 := 15.0),
                    y=(y_0 := 12.0),
                    heading=(theta_0 := 0.5),
                ),
                M,
                T,
                x_0,
                y_0,
                theta_0,
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "M", "T", "x_0", "y_0", "theta_0"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.unicycle),
            *cases(create_model=create_model.jax, data=data.jax.unicycle),
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
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(
            rollouts,
            array([[[x_0] * M, [y_0] * M, [theta_0] * M]] * T, shape=(T, 3, M)),
        )
        assert np.allclose(
            positions := rollouts.positions,
            array([[[x_0] * M, [y_0] * M]] * T, shape=(T, 2, M)),
        )
        assert np.allclose(positions.x(), array([[x_0] * M] * T, shape=(T, M)))
        assert np.allclose(positions.y(), array([[y_0] * M] * T, shape=(T, M)))
        assert np.allclose(rollouts.heading(), array([[theta_0] * M] * T, shape=(T, M)))


class test_that_vehicle_follows_straight_line_when_linear_velocity_is_constant_and_angular_is_zero:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.unicycle.dynamical(time_step_size=(dt := 0.25)),
                inputs := data.control_input_batch(
                    time_horizon=(T := 4),
                    rollout_count=(M := 3),
                    linear_velocity=(v := 4.47),
                    angular_velocity=0.0,
                ),
                initial_state := data.state(
                    x=(x_0 := 1.0),
                    y=(y_0 := 2.0),
                    heading=(theta_0 := np.arctan2(4, 2)),
                ),
                expected_x := array(
                    [
                        [x_0 + 1 * dt * v * np.cos(theta_0)] * M,
                        [x_0 + 2 * dt * v * np.cos(theta_0)] * M,
                        [x_0 + 3 * dt * v * np.cos(theta_0)] * M,
                        [x_0 + 4 * dt * v * np.cos(theta_0)] * M,
                    ],
                    shape=(T, M),
                ),
                expected_y := array(
                    [
                        [y_0 + 1 * dt * v * np.sin(theta_0)] * M,
                        [y_0 + 2 * dt * v * np.sin(theta_0)] * M,
                        [y_0 + 3 * dt * v * np.sin(theta_0)] * M,
                        [y_0 + 4 * dt * v * np.sin(theta_0)] * M,
                    ],
                    shape=(T, M),
                ),
                expected_theta := array([[theta_0] * M] * T, shape=(T, M)),
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
        ],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.unicycle),
            *cases(create_model=create_model.jax, data=data.jax.unicycle),
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
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts.positions.x(), expected_x)
        assert np.allclose(rollouts.positions.y(), expected_y)
        assert np.allclose(rollouts.heading(), expected_theta)


class test_that_heading_does_not_change_when_angular_velocity_is_zero:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.unicycle.dynamical(time_step_size=0.5),
                inputs := data.control_input_batch(
                    time_horizon=(T := 10),
                    rollout_count=(M := 4),
                    linear_velocity=2.5,
                    angular_velocity=0.0,
                ),
                initial_state := data.state(x=5.0, y=3.0, heading=(theta_0 := 1.23)),
                expected_theta := array([[theta_0] * M] * T, shape=(T, M)),
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_theta"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.unicycle),
            *cases(create_model=create_model.jax, data=data.jax.unicycle),
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
        expected_theta: Array,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts.heading(), expected_theta)


class test_that_vehicle_orientation_returns_to_start_when_angular_velocity_is_reversed:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.unicycle.dynamical(time_step_size=0.5),
                inputs := data.control_input_batch(
                    rollout_count=(M := 8),
                    linear_velocity=array([1.0] * (T := 6), shape=(T,)),
                    angular_velocity=array(
                        [0.2, 0.4, 1.0, -1.0, -0.4, -0.2], shape=(T,)
                    ),
                ),
                initial_state := data.state(x=2.0, y=4.0, heading=(theta_0 := 1.23)),
                expected_final_theta := array([theta_0] * M, shape=(M,)),
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_final_theta"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.unicycle),
            *cases(create_model=create_model.jax, data=data.jax.unicycle),
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


class test_that_vehicle_returns_to_start_when_completing_a_circle:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                # Full circle with unicycle model:
                # For 2π rotation: T * dt * omega = 2π
                # So omega = 2π / (T * dt)
                model := create_model.unicycle.dynamical(time_step_size=(dt := 0.1)),
                inputs := data.control_input_batch(
                    rollout_count=(M := 2),
                    linear_velocity=array([(v := 1.2)] * (T := 100), shape=(T,)),
                    angular_velocity=array(
                        [(omega := 2 * np.pi / (T * dt))] * T, shape=(T,)
                    ),
                ),
                initial_state := data.state(
                    x=(x_0 := 2.4), y=(y_0 := 3.6), heading=(theta_0 := 0.5)
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
            *cases(create_model=create_model.numpy, data=data.numpy.unicycle),
            *cases(create_model=create_model.jax, data=data.jax.unicycle),
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


class test_that_displacement_is_consistent_with_velocity_input:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.unicycle.dynamical(time_step_size=(dt := 0.1)),
                inputs := data.control_input_batch(
                    rollout_count=(M := 2),
                    linear_velocity=np.random.uniform(0.5, 2.0, size=(T := 12)),
                    angular_velocity=np.random.uniform(-0.1, 0.1, size=(T,)),
                ),
                initial_state := data.state(x=4.0, y=5.0, heading=1.2),
                time_step_size := dt,
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "time_step_size"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.unicycle),
            *cases(create_model=create_model.jax, data=data.jax.unicycle),
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

        # Actual displacements
        delta_x = np.diff(full_x, axis=0)
        delta_y = np.diff(full_y, axis=0)

        displacement_magnitudes = np.sqrt(delta_x**2 + delta_y**2)

        assert np.allclose(
            displacement_magnitudes,
            inputs.linear_velocity() * time_step_size,
            atol=1e-2,
        )


class test_that_speed_is_clamped_to_limits:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.unicycle.dynamical(
                    time_step_size=0.5,
                    speed_limits=(-5.0, 10.0),
                ),
                inputs := data.control_input_batch(
                    time_horizon=10,
                    rollout_count=3,
                    linear_velocity=15.0,
                    angular_velocity=0.0,
                ),
                initial_state := data.state(x=3.0, y=4.0, heading=0.0),
                expected_speed := 10.0,
            ),
            (
                model := create_model.unicycle.dynamical(
                    time_step_size=0.5,
                    speed_limits=(-5.0, 10.0),
                ),
                inputs := data.control_input_batch(
                    time_horizon=10,
                    rollout_count=3,
                    linear_velocity=-10.0,
                    angular_velocity=0.0,
                ),
                initial_state := data.state(x=3.0, y=4.0, heading=0.0),
                expected_speed := 5.0,
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_speed"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.unicycle),
            *cases(create_model=create_model.jax, data=data.jax.unicycle),
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
        expected_speed: float,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        x_with_initial = np.insert(rollouts.positions.x(), 0, initial_state.x, axis=0)
        y_with_initial = np.insert(rollouts.positions.y(), 0, initial_state.y, axis=0)

        delta_x = np.diff(x_with_initial, axis=0)
        delta_y = np.diff(y_with_initial, axis=0)

        displacement_magnitudes = np.sqrt(delta_x**2 + delta_y**2)
        actual_speed = displacement_magnitudes / model.time_step_size

        assert np.allclose(actual_speed, expected_speed, atol=1e-4)


class test_that_angular_velocity_is_clipped_to_limits:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            *[
                (
                    model := create_model.unicycle.dynamical(
                        time_step_size=(dt := 1.0),
                        angular_velocity_limits=(omega_min := -0.2, omega_max := 0.3),
                    ),
                    inputs := data.control_input_batch(
                        time_horizon=(T := 5),
                        rollout_count=(M := 4),
                        linear_velocity=0.5,
                        angular_velocity=omega,
                    ),
                    initial_state := data.state(x=2.0, y=4.0, heading=2.0),
                    expected_theta_change := array(
                        [(omega_max if expected == "max" else omega_min) * (dt * T)]
                        * M,
                        shape=(M,),
                    ),
                )
                for (omega, expected) in [(1.0, "max"), (-1.0, "min")]
            ],
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_theta_change"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy.unicycle),
            *cases(create_model=create_model.jax, data=data.jax.unicycle),
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


class test_that_simulating_individual_steps_matches_horizon_simulation:
    @staticmethod
    def cases(create_model, data, types) -> Sequence[tuple]:
        return [
            (
                model := create_model.unicycle.dynamical(
                    time_step_size=(dt := 0.1),
                    speed_limits=(0.0, 20.0),
                    angular_velocity_limits=(-0.4, 0.4),
                ),
                input_batch := data.control_input_batch(
                    rollout_count=1,
                    linear_velocity=array([2.0, 1.5, 0.5, 1.0, 3.0], shape=(T := 5,)),
                    angular_velocity=array([0.1, -0.1, 0.2, 0.0, -0.2], shape=(T,)),
                ),
                initial_state := data.state(x=0.0, y=0.0, heading=0.0),
                horizon := T,
                input_of := lambda input_batch, t: (
                    types.unicycle.control_input_sequence(input_batch.array[t:, :, 0])
                ),
            ),
        ]

    @mark.parametrize(
        ["model", "input_batch", "initial_state", "horizon", "input_of"],
        [
            *cases(
                create_model=create_model.numpy,
                data=data.numpy.unicycle,
                types=types.numpy,
            ),
            *cases(
                create_model=create_model.jax, data=data.jax.unicycle, types=types.jax
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
                model := create_model.unicycle.dynamical(
                    time_step_size=(dt := 0.1),
                    speed_limits=(0.0, 20.0),
                    angular_velocity_limits=(-0.4, 0.4),
                ),
                input_batch := data.control_input_batch(
                    rollout_count=1,
                    linear_velocity=array([1.0, 0.5, 2.0, 1.5], shape=(T := 4,)),
                    angular_velocity=array([0.1, -0.1, 0.2, 0.0], shape=(T,)),
                ),
                initial_state := data.state(x=0.0, y=0.0, heading=0.0),
                input_of := lambda input_batch, t: (
                    types.unicycle.control_input_sequence(input_batch.array[..., t])
                ),
            ),
        ]

    @mark.parametrize(
        ["model", "input_batch", "initial_state", "input_of"],
        [
            *cases(
                create_model=create_model.numpy,
                data=data.numpy.unicycle,
                types=types.numpy,
            ),
            *cases(
                create_model=create_model.jax, data=data.jax.unicycle, types=types.jax
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
