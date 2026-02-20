from typing import Callable, Sequence

from faran import (
    model as create_model,
    DynamicalModel,
    State,
    StateSequence,
    StateBatch,
)

from numtypes import array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


class test_that_integration_is_exact_for_constant_inputs:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.integrator.dynamical(time_step_size=(dt := 0.1)),
                inputs := data.control_input_batch(
                    array(
                        [[[(v_theta := 10.0)] * (M := 2)]] * (T := 1),
                        shape=(T, D_u := 1, M),
                    )
                ),
                initial_state := data.state(
                    array([(theta_0 := 5.0)], shape=(D_x := 1,))
                ),
                expected_states := array(
                    [[[theta_0 + v_theta * dt] * M]] * T, shape=(T, D_x, M)
                ),
            ),
            (
                model := create_model.integrator.dynamical(time_step_size=(dt := 0.1)),
                inputs := data.control_input_batch(
                    array(
                        [[[(v_theta := 10.0)] * (M := 1)]] * (T := 5),
                        shape=(T, D_u := 1, M),
                    )
                ),
                initial_state := data.state(
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
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_states"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy),
            *cases(create_model=create_model.jax, data=data.jax),
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
        expected_states: StateBatchT,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts, expected_states)


class test_that_zero_velocity_results_in_standstill:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.integrator.dynamical(time_step_size=(dt := 0.1)),
                inputs := data.control_input_batch(
                    array(
                        [[[(v_theta := 0.0)] * (M := 1)]] * (T := 100),
                        shape=(T, D_u := 1, M),
                    )
                ),
                initial_state := data.state(
                    array([(theta_0 := 5.0)], shape=(D_x := 1,))
                ),
                expected_states := data.state_batch(np.full((T, D_x, M), theta_0)),
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_states"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy),
            *cases(create_model=create_model.jax, data=data.jax),
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
        expected_states: StateBatchT,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts, expected_states)


class test_that_state_limits_are_respected_during_integration:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.integrator.dynamical(
                    time_step_size=(dt := 0.1),
                    state_limits=((state_min := -1.0), (state_max := 10.0)),
                ),
                inputs := data.control_input_batch(
                    array(
                        [[[(v_theta := 5.0)] * (M := 1)]] * (T := 30),
                        shape=(T, D_u := 1, M),
                    )
                ),
                initial_state := data.state(
                    array([(theta_0 := 8.0)], shape=(D_x := 1,))
                ),
                state_min,
                state_max,
                # Without limits: 8.0 + 30*0.5 = 23.0
                # With limits: clipped to 10.0 and stays there
                expected_final_state := array([[state_max] * M], shape=(D_x, M)),
            ),
            (
                model := create_model.integrator.dynamical(
                    time_step_size=(dt := 0.1),
                    state_limits=((state_min := 2.0), (state_max := 10.0)),
                ),
                inputs := data.control_input_batch(
                    array(
                        [[[(v_theta := -5.0)] * (M := 1)]] * (T := 30),
                        shape=(T, D_u := 1, M),
                    )
                ),
                initial_state := data.state(
                    array([(theta_0 := 2.0)], shape=(D_x := 1,))
                ),
                state_min,
                state_max,
                # Without limits: 2.0 - 30*0.5 = -13.0
                # With limits: clipped to 2.0 and stays there
                expected_final_state := array([[state_min] * M], shape=(D_x, M)),
            ),
        ]

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
            *cases(create_model=create_model.numpy, data=data.numpy),
            *cases(create_model=create_model.jax, data=data.jax),
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
        state_min: float,
        state_max: float,
        expected_final_state: StateBatchT,
    ) -> None:
        rollouts = np.asarray(model.simulate(inputs, initial_state))

        assert np.all(rollouts >= state_min)
        assert np.all(rollouts <= state_max)

        assert np.allclose(rollouts[-1], expected_final_state)


class test_that_velocity_limits_are_respected_during_integration:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.integrator.dynamical(
                    time_step_size=(dt := 0.1),
                    velocity_limits=((v_min := -1.0), (v_max := 2.0)),
                ),
                # Velocity of 10.0 exceeds limit of 2.0
                inputs := data.control_input_batch(
                    array(
                        [[[(v_theta := 10.0)] * (M := 1)]] * (T := 5),
                        shape=(T, D_u := 1, M),
                    )
                ),
                initial_state := data.state(
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
                model := create_model.integrator.dynamical(
                    time_step_size=(dt := 0.1),
                    velocity_limits=((v_min := -2.0), (v_max := 4.0)),
                ),
                # Velocity of -10.0 exceeds limit of -2.0
                inputs := data.control_input_batch(
                    array(
                        [[[(v_theta := -10.0)] * (M := 1)]] * (T := 3),
                        shape=(T, D_u := 1, M),
                    )
                ),
                initial_state := data.state(
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
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_states"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy),
            *cases(create_model=create_model.jax, data=data.jax),
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
        expected_states: StateBatchT,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts, expected_states)


class test_that_state_wraps_around_when_periodic_boundaries_are_enabled:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return (
            [
                (
                    model := create_model.integrator.dynamical(
                        time_step_size=(dt := 0.1),
                        state_limits=((lower := 0.0), (upper := 10.0)),
                        periodic=True,
                    ),
                    inputs := data.control_input_batch(
                        array(
                            [[[(v_theta := 5.0)] * (M := 1)]] * (T := 1),
                            shape=(T, D_u := 1, M),
                        )
                    ),
                    initial_state := data.state(
                        array([(theta_0 := 9.9)], shape=(D_x := 1,))
                    ),
                    expected_states := array(
                        [
                            [
                                [
                                    lower
                                    + ((theta_0 + v_theta * dt) - lower)
                                    % (upper - lower)
                                ]
                            ]
                        ],
                        shape=(T, D_x, M),
                    ),
                ),
                (
                    model := create_model.integrator.dynamical(
                        time_step_size=(dt := 0.1),
                        state_limits=((lower := 0.0), (upper := 10.0)),
                        periodic=True,
                    ),
                    inputs := data.control_input_batch(
                        array(
                            [[[(v_theta := -5.0)] * (M := 1)]] * (T := 1),
                            shape=(T, D_u := 1, M),
                        )
                    ),
                    initial_state := data.state(
                        array([(theta_0 := 0.2)], shape=(D_x := 1,))
                    ),
                    expected_states := array(
                        [
                            [
                                [
                                    lower
                                    + ((theta_0 + v_theta * dt) - lower)
                                    % (upper - lower)
                                ]
                            ]
                        ],
                        shape=(T, D_x, M),
                    ),
                ),
                (
                    model := create_model.integrator.dynamical(
                        time_step_size=(dt := 0.1),
                        state_limits=((lower := 0.0), (upper := 10.0)),
                        periodic=True,
                    ),
                    inputs := data.control_input_batch(
                        array(
                            [[[(v_theta := 250.0)] * (M := 1)]] * (T := 1),
                            shape=(T, D_u := 1, M),
                        )
                    ),
                    initial_state := data.state(
                        array([(theta_0 := 1.0)], shape=(D_x := 1,))
                    ),
                    expected_states := array(
                        [
                            [
                                [
                                    lower
                                    + ((theta_0 + v_theta * dt) - lower)
                                    % (upper - lower)
                                ]
                            ]
                        ],
                        shape=(T, D_x, M),
                    ),
                ),
                (
                    model := create_model.integrator.dynamical(
                        time_step_size=(dt := 0.1),
                        state_limits=((lower := 0.0), (upper := 10.0)),
                        periodic=True,
                    ),
                    inputs := data.control_input_batch(
                        array(
                            [[[(v_theta := 15.0)] * (M := 1)]] * (T := 5),
                            shape=(T, D_u := 1, M),
                        )
                    ),
                    initial_state := data.state(
                        array([(theta_0 := 9.0)], shape=(D_x := 1,))
                    ),
                    expected_states := array(
                        [
                            [
                                [
                                    lower
                                    + ((theta_0 + v_theta * dt * 1) - lower)
                                    % (upper - lower)
                                ]
                            ],
                            [
                                [
                                    lower
                                    + ((theta_0 + v_theta * dt * 2) - lower)
                                    % (upper - lower)
                                ]
                            ],
                            [
                                [
                                    lower
                                    + ((theta_0 + v_theta * dt * 3) - lower)
                                    % (upper - lower)
                                ]
                            ],
                            [
                                [
                                    lower
                                    + ((theta_0 + v_theta * dt * 4) - lower)
                                    % (upper - lower)
                                ]
                            ],
                            [
                                [
                                    lower
                                    + ((theta_0 + v_theta * dt * 5) - lower)
                                    % (upper - lower)
                                ]
                            ],
                        ],
                        shape=(T, D_x, M),
                    ),
                ),
            ]
            + [  # Negative state limits
                (
                    model := create_model.integrator.dynamical(
                        time_step_size=(dt := 0.1),
                        state_limits=((lower := -np.pi), (upper := np.pi)),
                        periodic=True,
                    ),
                    inputs := data.control_input_batch(
                        array(
                            [[[(v_theta := 1.0)] * (M := 1)]] * (T := 1),
                            shape=(T, D_u := 1, M),
                        )
                    ),
                    initial_state := data.state(
                        array([(theta_0 := upper - 0.05)], shape=(D_x := 1,))
                    ),
                    expected_states := array(
                        [
                            [
                                [
                                    lower
                                    + ((theta_0 + v_theta * dt) - lower)
                                    % (upper - lower)
                                ]
                            ]
                        ],
                        shape=(T, D_x, M),
                    ),
                ),
                (
                    model := create_model.integrator.dynamical(
                        time_step_size=(dt := 0.1),
                        state_limits=((lower := -np.pi), (upper := np.pi)),
                        periodic=True,
                    ),
                    inputs := data.control_input_batch(
                        array(
                            [[[(v_theta := -1.0)] * (M := 1)]] * (T := 1),
                            shape=(T, D_u := 1, M),
                        )
                    ),
                    initial_state := data.state(
                        array([(theta_0 := lower + 0.02)], shape=(D_x := 1,))
                    ),
                    expected_states := array(
                        [
                            [
                                [
                                    lower
                                    + ((theta_0 + v_theta * dt) - lower)
                                    % (upper - lower)
                                ]
                            ]
                        ],
                        shape=(T, D_x, M),
                    ),
                ),
            ]
            + [  # Multiple rollouts
                (
                    model := create_model.integrator.dynamical(
                        time_step_size=(dt := 1.0),
                        state_limits=((lower := 0.0), (upper := 10.0)),
                        periodic=True,
                    ),
                    inputs := data.control_input_batch(
                        array(
                            [[[2.0, 20.0, -5.0]]],
                            shape=(T := 1, D_u := 1, M := 3),
                        )
                    ),
                    initial_state := data.state(
                        array([(theta_0 := 9.0)], shape=(D_x := 1,))
                    ),
                    expected_states := array(
                        [
                            [
                                [
                                    lower
                                    + ((theta_0 + 2.0 * dt) - lower) % (upper - lower),
                                    lower
                                    + ((theta_0 + 20.0 * dt) - lower) % (upper - lower),
                                    lower
                                    + ((theta_0 + (-5.0) * dt) - lower)
                                    % (upper - lower),
                                ]
                            ]
                        ],
                        shape=(T, D_x, M),
                    ),
                ),
            ]
        )

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_states"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy),
            *cases(create_model=create_model.jax, data=data.jax),
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
        expected_states: StateBatchT,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts, expected_states)


class test_that_velocity_limits_are_applied_before_periodic_wrapping:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.integrator.dynamical(
                    time_step_size=(dt := 1.0),
                    state_limits=((lower := 0.0), (upper := 10.0)),
                    velocity_limits=((v_min := -1.0), (v_max := 2.0)),
                    periodic=True,
                ),
                inputs := data.control_input_batch(
                    array(
                        [[[(v_theta := 10.0)] * (M := 1)]] * (T := 1),
                        shape=(T, D_u := 1, M),
                    )
                ),
                initial_state := data.state(
                    array([(theta_0 := 9.0)], shape=(D_x := 1,))
                ),
                expected_states := array(
                    [[[lower + ((theta_0 + v_max * dt) - lower) % (upper - lower)]]],
                    shape=(T, D_x, M),
                ),
            ),
            (
                model := create_model.integrator.dynamical(
                    time_step_size=(dt := 1.0),
                    state_limits=((lower := 0.0), (upper := 10.0)),
                    velocity_limits=((v_min := -1.0), (v_max := 2.0)),
                    periodic=True,
                ),
                inputs := data.control_input_batch(
                    array(
                        [[[(v_theta := -10.0)] * (M := 1)]] * (T := 1),
                        shape=(T, D_u := 1, M),
                    )
                ),
                initial_state := data.state(
                    array([(theta_0 := 0.5)], shape=(D_x := 1,))
                ),
                expected_states := array(
                    [[[lower + ((theta_0 + v_min * dt) - lower) % (upper - lower)]]],
                    shape=(T, D_x, M),
                ),
            ),
        ]

    @mark.parametrize(
        ["model", "inputs", "initial_state", "expected_states"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy),
            *cases(create_model=create_model.jax, data=data.jax),
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
        expected_states: StateBatchT,
    ) -> None:
        rollouts = model.simulate(inputs, initial_state)

        assert np.allclose(rollouts, expected_states)


class test_that_simulating_individual_steps_matches_horizon_simulation:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        return [
            (
                model := create_model.integrator.dynamical(
                    time_step_size=0.1,
                    state_limits=(-5.0, 5.0),
                    velocity_limits=(-1.0, 2.0),
                ),
                input_batch := data.control_input_batch(
                    np.random.uniform(-3.0, 3.0, size=(T := 4, D_x := 3, M := 1))
                ),
                initial_state := data.state(np.random.uniform(-2.0, 2.0, size=(D_x,))),
                horizon := T,
                input_of := (
                    lambda input_batch, t: data.control_input_sequence(
                        input_batch.array[t:, :, 0]
                    )
                ),
            ),
            (
                model := create_model.integrator.dynamical(
                    time_step_size=1.0,
                    state_limits=(-np.pi, np.pi),
                    velocity_limits=(-10.0, 10.0),
                    periodic=True,
                ),
                input_batch := data.control_input_batch(
                    np.full((T := 4, D_x := 2, M := 1), 3.0)
                ),
                initial_state := data.state(np.zeros((D_x,))),
                horizon := T,
                input_of := (
                    lambda input_batch, t: data.control_input_sequence(
                        input_batch.array[t:, :, 0]
                    )
                ),
            ),
        ]

    @mark.parametrize(
        ["model", "input_batch", "initial_state", "horizon", "input_of"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy),
            *cases(create_model=create_model.jax, data=data.jax),
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
        rollout = np.asarray(model.simulate(input_batch, initial_state)).squeeze()
        current_state = initial_state

        for t in range(horizon):
            current_state = model.step(input_of(input_batch, t), current_state)

            assert np.allclose(current_state, rollout[t], atol=1e-6), (
                f"Mismatch at time step {t}: expected {rollout[t]}, got {current_state}"
            )


class test_that_simulating_individual_input_sequence_matches_horizon_simulation:
    @staticmethod
    def cases(create_model, data) -> Sequence[tuple]:
        # NOTE: Make sure to use M=1.
        M = 1

        return [
            (
                model := create_model.integrator.dynamical(
                    time_step_size=(dt := 0.2),
                    state_limits=(-20.0, 20.0),
                    velocity_limits=(-5.0, 5.0),
                ),
                input_batch := data.control_input_batch(
                    np.random.uniform(-10.0, 10.0, size=(T := 6, D_x := 4, M))
                ),
                initial_state := data.state(
                    np.random.uniform(-10.0, 10.0, size=(D_x,))
                ),
                input_at := (
                    lambda input_batch, m: data.control_input_sequence(
                        input_batch.array[..., m]
                    )
                ),
            ),
            (
                model := create_model.integrator.dynamical(
                    time_step_size=(dt := 0.5),
                    state_limits=(-np.pi, np.pi),
                    velocity_limits=(-15.0, 15.0),
                    periodic=True,
                ),
                input_batch := data.control_input_batch(
                    np.full((T := 3, D_x := 2, M), -4.0)
                ),
                initial_state := data.state(np.zeros((D_x,))),
                input_at := (
                    lambda input_batch, m: data.control_input_sequence(
                        input_batch.array[..., m]
                    )
                ),
            ),
        ]

    @mark.parametrize(
        ["model", "input_batch", "initial_state", "input_of"],
        [
            *cases(create_model=create_model.numpy, data=data.numpy),
            *cases(create_model=create_model.jax, data=data.jax),
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
        rollout = np.asarray(model.simulate(input_batch, initial_state)).squeeze()
        sequence = model.forward(input_of(input_batch, 0), initial_state)

        assert np.allclose(sequence, rollout, atol=1e-6)
