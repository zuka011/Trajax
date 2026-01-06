from typing import Callable

from trajax import (
    model as create_model,
    DynamicalModel,
    State,
    StateSequence,
    StateBatch,
)

from numtypes import array

import numpy as np

from tests.dsl import mppi as data, clear_type
from pytest import mark


@mark.parametrize(
    ["model", "inputs", "initial_state", "expected_states"],
    [
        (
            model := create_model.numpy.integrator.dynamical(
                time_step_size=(dt := 0.1)
            ),
            inputs := data.numpy.control_input_batch(
                array(
                    [[[(v_theta := 10.0)] * (M := 2)]] * (T := 1),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.numpy.state(
                array([(theta_0 := 5.0)], shape=(D_x := 1,))
            ),
            expected_states := array(
                [[[theta_0 + v_theta * dt] * M]] * T, shape=(T, D_x, M)
            ),
        ),
        (
            model := create_model.numpy.integrator.dynamical(
                time_step_size=(dt := 0.1)
            ),
            inputs := data.numpy.control_input_batch(
                array(
                    [[[(v_theta := 10.0)] * (M := 1)]] * (T := 5),
                    shape=(T, D_u := 1, M),  # type: ignore
                )
            ),
            initial_state := data.numpy.state(
                array([(theta_0 := 5.0)], shape=(D_x := 1,))
            ),
            # Using the formula: theta_t = theta_0 + v_theta * t
            expected_states := array(
                [
                    [[(theta_0 + v_theta * dt * 1)]],
                    [[(theta_0 + v_theta * dt * 2)]],
                    [[(theta_0 + v_theta * dt * 3)]],
                    [[(theta_0 + v_theta * dt * 4)]],
                    [[(theta_0 + v_theta * dt * 5)]],
                ],
                shape=(T, D_x, M),
            ),
        ),
        (
            model := create_model.jax.integrator.dynamical(time_step_size=(dt := 0.1)),
            inputs := data.jax.control_input_batch(
                array(
                    [[[(v_theta := 10.0)] * (M := 4)]] * (T := 1),
                    shape=(T, D_u := 1, M),  # type: ignore
                )
            ),
            initial_state := data.jax.state(
                array([(theta_0 := 5.0)], shape=(D_x := 1,))
            ),
            expected_final_state := array(
                [[[theta_0 + v_theta * dt] * M]] * T, shape=(T, D_x, M)
            ),
        ),
        (
            model := create_model.jax.integrator.dynamical(time_step_size=(dt := 0.1)),
            inputs := data.jax.control_input_batch(
                array(
                    [[[(v_theta := 10.0)] * (M := 1)]] * (T := 4),
                    shape=(T, D_u := 1, M),  # type: ignore
                )
            ),
            initial_state := data.jax.state(
                array([(theta_0 := 5.0)], shape=(D_x := 1,))
            ),
            expected_states := array(
                [
                    [[(theta_0 + v_theta * dt * 1)]],
                    [[(theta_0 + v_theta * dt * 2)]],
                    [[(theta_0 + v_theta * dt * 3)]],
                    [[(theta_0 + v_theta * dt * 4)]],
                ],
                shape=(T, D_x, M),
            ),
        ),
    ],
)
def test_that_integration_is_exact_for_constant_inputs[
    StateT,
    StateSequenceT,
    StateBatchT: StateBatch,
    ControlInputSequenceT,
    ControlInputBatchT,
](
    model: DynamicalModel[
        StateT, StateSequenceT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_states: StateBatchT,
) -> None:
    rollouts = model.simulate(inputs, initial_state)

    assert np.allclose(rollouts, expected_states)


T = clear_type
D_u = clear_type
D_x = clear_type
M = clear_type


@mark.parametrize(
    ["model", "inputs", "initial_state", "expected_states"],
    [
        (
            model := create_model.numpy.integrator.dynamical(
                time_step_size=(dt := 0.1)
            ),
            inputs := data.numpy.control_input_batch(
                array(
                    [[[(v_theta := 0.0)] * (M := 1)]] * (T := 100),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.numpy.state(
                array([(theta_0 := 5.0)], shape=(D_x := 1,))
            ),
            expected_states := data.numpy.state_batch(
                np.full((T, D_x, M), theta_0)  # type: ignore
            ),
        ),
        (
            model := create_model.jax.integrator.dynamical(time_step_size=(dt := 0.1)),
            inputs := data.jax.control_input_batch(
                array(
                    [[[(v_theta := 0.0)] * (M := 1)]] * (T := 100),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.jax.state(
                array([(theta_0 := 5.0)], shape=(D_x := 1,))
            ),
            expected_states := data.jax.state_batch(
                np.full((T, D_x, M), theta_0)  # type: ignore
            ),
        ),
    ],
)
def test_that_zero_velocity_results_in_standstill[
    StateT,
    StateSequenceT,
    StateBatchT: StateBatch,
    ControlInputSequenceT,
    ControlInputBatchT,
](
    model: DynamicalModel[
        StateT, StateSequenceT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_states: StateBatchT,
) -> None:
    rollouts = model.simulate(inputs, initial_state)

    assert np.allclose(rollouts, expected_states)


@mark.parametrize(
    [
        "model",
        "inputs",
        "initial_state",
        "state_min",
        "state_max",
        "expected_final_state",
    ],
    [
        (
            model := create_model.numpy.integrator.dynamical(
                time_step_size=(dt := 0.1),
                state_limits=((state_min := -1.0), (state_max := 10.0)),
            ),
            inputs := data.numpy.control_input_batch(
                array(
                    [[[(v_theta := 5.0)] * (M := 1)]] * (T := 30),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.numpy.state(
                array([(theta_0 := 8.0)], shape=(D_x := 1,))
            ),
            state_min,
            state_max,
            # Without limits: 8.0 + 30*0.5 = 23.0
            # With limits: clipped to 10.0 and stays there
            expected_final_state := array([[state_max] * M], shape=(D_x, M)),
        ),
        (
            model := create_model.numpy.integrator.dynamical(
                time_step_size=(dt := 0.1),
                state_limits=((state_min := 2.0), (state_max := 10.0)),
            ),
            inputs := data.numpy.control_input_batch(
                array(
                    [[[(v_theta := -5.0)] * (M := 1)]] * (T := 30),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.numpy.state(
                array([(theta_0 := 2.0)], shape=(D_x := 1,))
            ),
            state_min,
            state_max,
            # Without limits: 2.0 - 30*0.5 = -13.0
            # With limits: clipped to 2.0 and stays there
            expected_final_state := array([[state_min] * M], shape=(D_x, M)),
        ),
        (
            model := create_model.jax.integrator.dynamical(
                time_step_size=(dt := 0.1),
                state_limits=((state_min := 1.0), (state_max := 10.0)),
            ),
            inputs := data.jax.control_input_batch(
                array(
                    [[[(v_theta := 5.0)] * (M := 1)]] * (T := 30),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.jax.state(
                array([(theta_0 := 8.0)], shape=(D_x := 1,))
            ),
            state_min,
            state_max,
            expected_final_state := array([[state_max] * M], shape=(D_x, M)),
        ),
        (
            model := create_model.jax.integrator.dynamical(
                time_step_size=(dt := 0.1),
                state_limits=((state_min := -4.0), (state_max := 10.0)),
            ),
            inputs := data.jax.control_input_batch(
                array(
                    [[[(v_theta := -5.0)] * (M := 1)]] * (T := 30),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.jax.state(
                array([(theta_0 := 2.0)], shape=(D_x := 1,))
            ),
            state_min,
            state_max,
            expected_final_state := array([[state_min] * M], shape=(D_x, M)),
        ),
    ],
)
def test_that_state_limits_are_respected_during_integration[
    StateT,
    StateSequenceT,
    StateBatchT: StateBatch,
    ControlInputSequenceT,
    ControlInputBatchT,
](
    model: DynamicalModel[
        StateT, StateSequenceT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    state_min: float,
    state_max: float,
    expected_final_state: StateBatchT,
) -> None:
    rollouts = np.asarray(model.simulate(inputs, initial_state))

    assert np.all(rollouts >= state_min)
    assert np.all(rollouts <= state_max)

    assert np.allclose(rollouts[-1], expected_final_state)


@mark.parametrize(
    ["model", "inputs", "initial_state", "expected_states"],
    [
        (
            model := create_model.numpy.integrator.dynamical(
                time_step_size=(dt := 0.1),
                velocity_limits=((v_min := -1.0), (v_max := 2.0)),
            ),
            # Velocity of 10.0 exceeds limit of 2.0
            inputs := data.numpy.control_input_batch(
                array(
                    [[[(v_theta := 10.0)] * (M := 1)]] * (T := 5),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.numpy.state(
                array([(theta_0 := 0.0)], shape=(D_x := 1,))
            ),
            # Velocity clipped to 2.0, so state = 0 + t*2.0*0.1
            expected_states := array(
                [
                    [[(theta_0 + v_max * dt * 1)]],
                    [[(theta_0 + v_max * dt * 2)]],
                    [[(theta_0 + v_max * dt * 3)]],
                    [[(theta_0 + v_max * dt * 4)]],
                    [[(theta_0 + v_max * dt * 5)]],
                ],
                shape=(T, D_x, M),
            ),
        ),
        (
            model := create_model.numpy.integrator.dynamical(
                time_step_size=(dt := 0.1),
                velocity_limits=((v_min := -2.0), (v_max := 4.0)),
            ),
            # Velocity of -10.0 exceeds limit of -2.0
            inputs := data.numpy.control_input_batch(
                array(
                    [[[(v_theta := -10.0)] * (M := 1)]] * (T := 3),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.numpy.state(
                array([(theta_0 := 5.0)], shape=(D_x := 1,))
            ),
            # Velocity clipped to -2.0, so state = 5 - t*2.0*0.1
            expected_states := array(
                [
                    [[(theta_0 + v_min * dt * 1)]],
                    [[(theta_0 + v_min * dt * 2)]],
                    [[(theta_0 + v_min * dt * 3)]],
                ],
                shape=(T, D_x, M),
            ),
        ),
        (
            model := create_model.jax.integrator.dynamical(
                time_step_size=(dt := 0.1),
                velocity_limits=((v_min := -4.0), (v_max := 2.0)),
            ),
            inputs := data.jax.control_input_batch(
                array(
                    [[[(v_theta := 10.0)] * (M := 1)]] * (T := 5),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.jax.state(
                array([(theta_0 := 0.0)], shape=(D_x := 1,))
            ),
            expected_states := array(
                [
                    [[(theta_0 + v_max * dt * 1)]],
                    [[(theta_0 + v_max * dt * 2)]],
                    [[(theta_0 + v_max * dt * 3)]],
                    [[(theta_0 + v_max * dt * 4)]],
                    [[(theta_0 + v_max * dt * 5)]],
                ],
                shape=(T, D_x, M),
            ),
        ),
        (
            model := create_model.jax.integrator.dynamical(
                time_step_size=(dt := 0.1),
                velocity_limits=((v_min := -2.0), (v_max := 5.0)),
            ),
            inputs := data.jax.control_input_batch(
                array(
                    [[[(v_theta := -10.0)] * (M := 1)]] * (T := 4),
                    shape=(T, D_u := 1, M),
                )
            ),
            initial_state := data.jax.state(
                array([(theta_0 := 5.0)], shape=(D_x := 1,))
            ),
            expected_states := array(
                [
                    [[(theta_0 + v_min * dt * 1)]],
                    [[(theta_0 + v_min * dt * 2)]],
                    [[(theta_0 + v_min * dt * 3)]],
                    [[(theta_0 + v_min * dt * 4)]],
                ],
                shape=(T, D_x, M),
            ),
        ),
    ],
)
def test_that_velocity_limits_are_respected_during_integration[
    StateT,
    StateSequenceT,
    StateBatchT: StateBatch,
    ControlInputSequenceT,
    ControlInputBatchT,
](
    model: DynamicalModel[
        StateT, StateSequenceT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_states: StateBatchT,
) -> None:
    rollouts = model.simulate(inputs, initial_state)

    assert np.allclose(rollouts, expected_states)


@mark.parametrize(
    ["model", "input_batch", "initial_state", "horizon", "input_of"],
    [
        (
            model := create_model.numpy.integrator.dynamical(
                time_step_size=0.1,
                state_limits=(-5.0, 5.0),
                velocity_limits=(-1.0, 2.0),
            ),
            input_batch := data.numpy.control_input_batch(
                np.random.uniform(-3.0, 3.0, size=(T := 4, D_x := 3, M := 1))
            ),
            initial_state := data.numpy.state(
                np.random.uniform(-2.0, 2.0, size=(D_x,))  # type: ignore
            ),
            horizon := T,
            input_of := (
                lambda input_batch, t: data.numpy.control_input_sequence(
                    input_batch.array[t:, :, 0]
                )
            ),
        ),
        (
            model := create_model.jax.integrator.dynamical(
                time_step_size=0.2,
                state_limits=(-10.0, 10.0),
                velocity_limits=(-2.0, 3.0),
            ),
            input_batch := data.jax.control_input_batch(
                np.random.uniform(-5.0, 5.0, size=(T := 3, D_x := 2, M := 1))
            ),
            initial_state := data.jax.state(
                np.random.uniform(-3.0, 3.0, size=(D_x,))  # type: ignore
            ),
            horizon := T,
            input_of := (
                lambda input_batch, t: data.jax.control_input_sequence(
                    input_batch.array[t:, :, 0]
                )
            ),
        ),
    ],
)
def test_that_simulating_individual_steps_matches_horizon_simulation[
    StateT: State,
    StateSequenceT,
    StateBatchT: StateBatch,
    ControlInputSequenceT,
    ControlInputBatchT,
](
    model: DynamicalModel[
        StateT, StateSequenceT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ],
    input_batch: ControlInputBatchT,
    initial_state: StateT,
    horizon: int,
    input_of: Callable[[ControlInputBatchT, int], ControlInputSequenceT],
) -> None:
    rollout = np.asarray(model.simulate(input_batch, initial_state)).squeeze()
    current_state = initial_state

    for t in range(horizon):
        current_state = model.step(input_of(input_batch, t), current_state)

        assert np.allclose(current_state, rollout[t], atol=1e-6), (
            f"Mismatch at time step {t}: expected {rollout[t]}, got {current_state}"
        )


@mark.parametrize(
    ["model", "input_batch", "initial_state", "input_of"],
    [
        (
            model := create_model.numpy.integrator.dynamical(
                time_step_size=(dt := 0.2),
                state_limits=(-20.0, 20.0),
                velocity_limits=(-5.0, 5.0),
            ),
            input_batch := data.numpy.control_input_batch(
                np.random.uniform(-10.0, 10.0, size=(T := 6, D_x := 4, M := 1))
            ),
            initial_state := data.numpy.state(
                np.random.uniform(-10.0, 10.0, size=(D_x,))  # type: ignore
            ),
            input_at := (
                lambda input_batch, m: data.numpy.control_input_sequence(
                    input_batch.array[..., m]
                )
            ),
        ),
        (
            model := create_model.jax.integrator.dynamical(
                time_step_size=(dt := 0.3),
                state_limits=(-30.0, 30.0),
                velocity_limits=(-6.0, 6.0),
            ),
            input_batch := data.jax.control_input_batch(
                np.random.uniform(-15.0, 15.0, size=(T := 5, D_x := 2, M := 1))
            ),
            initial_state := data.jax.state(
                np.random.uniform(-15.0, 15.0, size=(D_x,))  # type: ignore
            ),
            input_at := (
                lambda input_batch, m: data.jax.control_input_sequence(
                    input_batch.array[..., m]
                )
            ),
        ),
    ],
)
def test_that_simulating_individual_input_sequence_matches_horizon_simulation[
    StateT,
    StateSequenceT: StateSequence,
    StateBatchT: StateBatch,
    ControlInputSequenceT,
    ControlInputBatchT,
](
    model: DynamicalModel[
        StateT, StateSequenceT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ],
    input_batch: ControlInputBatchT,
    initial_state: StateT,
    input_of: Callable[[ControlInputBatchT, int], ControlInputSequenceT],
) -> None:
    rollout = np.asarray(model.simulate(input_batch, initial_state)).squeeze()
    sequence = model.forward(input_of(input_batch, 0), initial_state)

    assert np.allclose(sequence, rollout, atol=1e-6)
