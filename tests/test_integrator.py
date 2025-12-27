from trajax import (
    model as create_model,
    IntegratorModel,
    State,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
)

from numtypes import array

import numpy as np

from tests.dsl import mppi as data, clear_type
from pytest import mark


@mark.parametrize(
    ["model", "inputs", "initial_state", "expected_states"],
    [
        (
            model := create_model.numpy.integrator(time_step_size=(dt := 0.1)),
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
            model := create_model.numpy.integrator(time_step_size=(dt := 0.1)),
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
            model := create_model.jax.integrator(time_step_size=(dt := 0.1)),
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
            model := create_model.jax.integrator(time_step_size=(dt := 0.1)),
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
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
](
    model: IntegratorModel[
        StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
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
            model := create_model.numpy.integrator(time_step_size=(dt := 0.1)),
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
            model := create_model.jax.integrator(time_step_size=(dt := 0.1)),
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
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
](
    model: IntegratorModel[
        StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
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
            model := create_model.numpy.integrator(
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
            model := create_model.numpy.integrator(
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
            model := create_model.jax.integrator(
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
            model := create_model.jax.integrator(
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
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
](
    model: IntegratorModel[
        StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
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
            model := create_model.numpy.integrator(
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
            model := create_model.numpy.integrator(
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
            model := create_model.jax.integrator(
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
            model := create_model.jax.integrator(
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
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
](
    model: IntegratorModel[
        StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ],
    inputs: ControlInputBatchT,
    initial_state: StateT,
    expected_states: StateBatchT,
) -> None:
    rollouts = model.simulate(inputs, initial_state)

    assert np.allclose(rollouts, expected_states)
