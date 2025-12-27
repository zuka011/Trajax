from trajax import (
    State,
    StateBatch,
    ControlInputBatch,
    Mppi,
    DynamicalModel,
    ControlInputSequence,
    mppi as create_mppi,
    model as create_model,
    update,
    padding,
    samplers,
)

import numpy as np
import jax.random as jrandom
from numtypes import array

from tests.dsl import mppi as data, costs, stubs, clear_type
from pytest import mark


@mark.parametrize(
    [
        "mppi",
        "temperature",
        "nominal_input",
        "initial_state",
        "expected_optimal_control",
        "tolerance",
    ],
    [
        (
            mppi := create_mppi.numpy.base(
                model=stubs.DynamicalModel.returns(
                    rollouts=data.numpy.state_batch(  # Doesn't matter for this test case.
                        np.random.uniform(
                            low=0.0, high=1.0, size=(T := 4, D_x := 2, M := 3)
                        )
                    ),
                    when_control_inputs_are=(
                        sampled_inputs := data.numpy.control_input_batch(
                            array(
                                np.array(
                                    [
                                        # Medium energy sample.
                                        np.random.uniform(
                                            low=-6.0, high=-5.0, size=(T, D_u := 3)
                                        ),
                                        # Spends the least energy, should be favored.
                                        optimal_control := (
                                            np.random.uniform(
                                                low=-1.0, high=1.0, size=(T, D_u)
                                            )
                                        ),
                                        # High energy sample.
                                        np.random.uniform(
                                            low=10.0, high=11.0, size=(T, D_u)
                                        ),
                                    ]
                                )
                                .transpose(1, 2, 0)
                                .tolist(),
                                shape=(T, D_u, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.numpy.state(
                            array([1.0, 2.0], shape=(D_x,))
                        )
                    ),
                ),
                cost_function=costs.numpy.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.numpy.control_input_sequence(
                            np.random.uniform(low=-1.0, high=1.0, size=(T, D_u))
                        )
                    ),
                ),
            ),
            temperature := 0.1,  # Low temperature, optimal should be close to best.
            nominal_input,
            initial_state,
            expected_optimal_control := data.numpy.control_input_sequence(
                optimal_control
            ),
            tolerance := 1e-3,  # Allow some tolerance, since temp. is not that small.
        ),
        (  # Analogous test case for JAX implementation.
            mppi := create_mppi.jax.base(
                model=stubs.DynamicalModel.returns(
                    rollouts=data.jax.state_batch(
                        np.random.uniform(
                            low=2.0, high=3.0, size=(T := 4, D_x := 2, M := 3)
                        )
                    ),
                    when_control_inputs_are=(
                        sampled_inputs := data.jax.control_input_batch(
                            array(
                                np.array(
                                    [
                                        # High energy sample.
                                        np.random.uniform(
                                            low=12.0, high=15.0, size=(T, D_u := 3)
                                        ),
                                        # Optimal control sample.
                                        optimal_control := (
                                            np.random.uniform(
                                                low=-2.0, high=2.0, size=(T, D_u)
                                            )
                                        ),
                                        # Medium energy sample.
                                        np.random.uniform(
                                            low=-10.0, high=-7.0, size=(T, D_u)
                                        ),
                                    ]
                                )
                                .transpose(1, 2, 0)
                                .tolist(),
                                shape=(T, D_u, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.jax.state(array([1.0, 2.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.jax.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.jax.control_input_sequence(
                            np.random.uniform(low=-3.0, high=4.0, size=(T, D_u))
                        )
                    ),
                ),
            ),
            temperature := 0.1,
            nominal_input,
            initial_state,
            expected_optimal_control := data.jax.control_input_sequence(
                optimal_control
            ),
            tolerance := 1e-3,
        ),
    ],
)
def test_that_mppi_favors_samples_with_lower_costs[
    StateT: State,
    ControlInputSequenceT: ControlInputSequence,
](
    mppi: Mppi[StateT, ControlInputSequenceT],
    temperature: float,
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    expected_optimal_control: ControlInputSequenceT,
    tolerance: float,
) -> None:
    control = mppi.step(
        temperature=temperature,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.allclose(control.optimal, expected_optimal_control, atol=tolerance)


T = clear_type


@mark.parametrize(
    ["mppi", "nominal_input", "initial_state", "expected_nominal_control"],
    [
        (
            mppi := create_mppi.numpy.base(
                planning_interval=3,
                update_function=update.numpy.no_update(),
                padding_function=padding.numpy.zero(),
                model=stubs.DynamicalModel.returns(
                    rollouts := data.numpy.state_batch(
                        np.zeros((T := 5, D_x := 1, M := 5))
                    ),
                    when_control_inputs_are=(
                        sampled_inputs := data.numpy.control_input_batch(
                            np.random.uniform(low=-1.0, high=1.0, size=(T, D_u := 2, M))  # type: ignore
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.numpy.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.numpy.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.numpy.control_input_sequence(
                            array(
                                [
                                    [1.0, 2.0],
                                    [3.0, 4.0],
                                    [5.0, 6.0],
                                    [7.0, 8.0],
                                    [9.0, 10.0],
                                ],
                                shape=(T, D_u),
                            )
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
            # After one step, nominal control should be shifted left and padded with zeros.
            expected_nominal_control := data.numpy.control_input_sequence(
                array(
                    [
                        [7.0, 8.0],
                        [9.0, 10.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    shape=(T, D_u),
                )
            ),
        ),
        (
            mppi := create_mppi.jax.base(
                planning_interval=4,
                update_function=update.jax.no_update(),
                padding_function=padding.jax.zero(),
                model=stubs.DynamicalModel.returns(
                    rollouts := data.jax.state_batch(
                        np.zeros((T := 6, D_x := 1, M := 4))
                    ),
                    when_control_inputs_are=(
                        sampled_inputs_jax := data.jax.control_input_batch(
                            np.random.uniform(low=-1.0, high=1.0, size=(T, D_u := 2, M))  # type: ignore
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.jax.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.jax.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs_jax,
                    when_nominal_input_is=(
                        nominal_input := data.jax.control_input_sequence(
                            array(
                                [
                                    [1.0, 2.0],
                                    [3.0, 4.0],
                                    [5.0, 6.0],
                                    [7.0, 8.0],
                                    [9.0, 10.0],
                                    [11.0, 12.0],
                                ],
                                shape=(T, D_u),
                            )
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
            expected_nominal_control := data.jax.control_input_sequence(
                array(
                    [
                        [9.0, 10.0],
                        [11.0, 12.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    shape=(T, D_u),
                )
            ),
        ),
    ],
)
def test_that_mppi_shifts_control_sequence_left_when_replanning[
    StateT: State,
    ControlInputSequenceT: ControlInputSequence,
](
    mppi: Mppi[StateT, ControlInputSequenceT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    expected_nominal_control: ControlInputSequenceT,
) -> None:
    control = mppi.step(
        temperature=1.0,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.allclose(control.nominal, expected_nominal_control)


sampled_inputs = clear_type
nominal_input = clear_type


@mark.parametrize(
    ["mppi", "nominal_input", "initial_state", "expected_optimal_control"],
    [
        (
            mppi := create_mppi.numpy.base(
                model=stubs.DynamicalModel.returns(
                    rollouts := data.numpy.state_batch(
                        np.zeros((T := 2, D_x := 1, M := 3))
                    ),
                    when_control_inputs_are=(
                        sampled_inputs := data.numpy.control_input_batch(
                            array(
                                np.array(
                                    [
                                        # High cost sample
                                        control_1 := [[10.0, 10.0], [10.0, 10.0]],
                                        # Low cost sample
                                        control_2 := [[1.0, 1.0], [1.0, 1.0]],
                                        # Medium cost sample
                                        control_3 := [[5.0, 5.0], [5.0, 5.0]],
                                    ],
                                )
                                .transpose(1, 2, 0)
                                .tolist(),
                                shape=(T, D_u := 2, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.numpy.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.numpy.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.numpy.control_input_sequence(
                            array([[0.0, 0.0], [0.0, 0.0]], shape=(T, D_u))
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
            # Result should be close to average of all controls.
            expected_optimal_control := data.numpy.control_input_sequence(
                array(
                    np.mean([control_1, control_2, control_3], axis=0), shape=(T, D_u)
                )
            ),
        ),
        (
            mppi := create_mppi.jax.base(
                model=stubs.DynamicalModel.returns(
                    rollouts := data.jax.state_batch(
                        np.zeros((T := 2, D_x := 1, M := 3))
                    ),
                    when_control_inputs_are=(
                        sampled_inputs_jax := data.jax.control_input_batch(
                            array(
                                np.array(
                                    [
                                        # High cost sample
                                        control_1 := [[10.0, 10.0], [10.0, 10.0]],
                                        # Low cost sample
                                        control_2 := [[1.0, 1.0], [1.0, 1.0]],
                                        # Medium cost sample
                                        control_3 := [[5.0, 5.0], [5.0, 5.0]],
                                    ],
                                )
                                .transpose(1, 2, 0)
                                .tolist(),
                                shape=(T, D_u := 2, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.jax.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.jax.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs_jax,
                    when_nominal_input_is=(
                        nominal_input := data.jax.control_input_sequence(
                            array([[0.0, 0.0], [0.0, 0.0]], shape=(T, D_u))
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
            expected_optimal_control := data.jax.control_input_sequence(
                array(
                    np.mean([control_1, control_2, control_3], axis=0), shape=(T, D_u)
                )
            ),
        ),
    ],
)
def test_that_mppi_uses_samples_with_higher_costs_when_temperature_is_high[
    StateT: State,
    ControlInputSequenceT: ControlInputSequence,
](
    mppi: Mppi[StateT, ControlInputSequenceT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    expected_optimal_control: ControlInputSequenceT,
) -> None:
    control = mppi.step(
        temperature=3e3,  # High temperature, should use all samples equally.
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.allclose(control.optimal, expected_optimal_control, rtol=0.1)


@mark.parametrize(
    ["mppi", "nominal_input", "initial_state", "sampled_inputs"],
    [
        (
            # If the weights sum to 1, then the optimal control should be a convex combination
            # of the samples, meaning it must be within the convex hull of the samples.
            mppi := create_mppi.numpy.base(
                model=stubs.DynamicalModel.returns(
                    rollouts := data.numpy.state_batch(
                        np.zeros((T := 2, D_x := 1, M := 2))
                    ),
                    when_control_inputs_are=(
                        sampled_inputs := data.numpy.control_input_batch(
                            array(
                                np.array(
                                    [[[3.0], [3.0]], [[-3.0], [-3.0]]],
                                )
                                .transpose(1, 2, 0)
                                .tolist(),
                                shape=(T, D_u := 1, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.numpy.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.numpy.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.numpy.control_input_sequence(
                            array([[0.0], [0.0]], shape=(T, D_u))
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
            sampled_inputs,
        ),
        (
            mppi := create_mppi.jax.base(
                model=stubs.DynamicalModel.returns(
                    rollouts := data.jax.state_batch(
                        np.zeros((T := 2, D_x := 1, M := 2))
                    ),
                    when_control_inputs_are=(
                        sampled_inputs := data.jax.control_input_batch(
                            array(
                                np.array(
                                    [[[3.0], [3.0]], [[-3.0], [-3.0]]],
                                )
                                .transpose(1, 2, 0)
                                .tolist(),
                                shape=(T, D_u := 1, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.jax.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.jax.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.jax.control_input_sequence(
                            array([[0.0], [0.0]], shape=(T, D_u))
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
            sampled_inputs,
        ),
    ],
)
def test_that_mppi_optimal_control_is_convex_combination_of_samples[
    StateT: State,
    ControlInputSequenceT: ControlInputSequence,
](
    mppi: Mppi[StateT, ControlInputSequenceT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    sampled_inputs: ControlInputBatch,
) -> None:
    control = mppi.step(
        temperature=1.0,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    samples_array = np.asarray(sampled_inputs)
    optimal_array = np.asarray(control.optimal)

    sample_min = np.min(samples_array, axis=2)
    sample_max = np.max(samples_array, axis=2)

    assert np.all(optimal_array >= sample_min - 1e-6)
    assert np.all(optimal_array <= sample_max + 1e-6)


@mark.parametrize(
    [
        "mppi",
        "model",
        "current_state",
        "nominal_input",
        "temperature",
        "max_iterations",
        "convergence_threshold",
    ],
    [
        (
            mppi := create_mppi.numpy.base(
                model=(model := create_model.numpy.integrator(time_step_size=0.1)),
                cost_function=costs.numpy.quadratic_distance_to_origin(),
                sampler=samplers.sampler.numpy.gaussian(
                    standard_deviation=array([5.0, 5.0], shape=(D_u := 2,)),
                    rollout_count=200,
                    to_batch=data.numpy.control_input_batch,
                    seed=42,
                ),
            ),
            model,
            current_state := data.numpy.state(array([224.0, 52.0], shape=(D_x := 2,))),
            nominal_input := data.numpy.control_input_sequence(
                np.zeros((T := 10, D_u))
            ),
            temperature := 1.0,
            max_iterations := 100,
            convergence_threshold := 0.1,
        ),
        (
            mppi := create_mppi.jax.base(
                model=(model := create_model.jax.integrator(time_step_size=0.1)),
                cost_function=costs.jax.quadratic_distance_to_origin(),
                sampler=samplers.sampler.jax.gaussian(
                    standard_deviation=array([5.0, 5.0], shape=(D_u := 2,)),
                    rollout_count=200,
                    to_batch=data.jax.control_input_batch,
                    key=jrandom.PRNGKey(42),
                ),
            ),
            model,
            current_state := data.jax.state(array([45.0, 125.0], shape=(D_x := 2,))),
            nominal_input := data.jax.control_input_sequence(np.zeros((T := 10, D_u))),
            temperature := 1.0,
            max_iterations := 100,
            convergence_threshold := 0.1,
        ),
    ],
)
def test_that_mppi_converges_to_target_state[
    StateT: State,
    ControlInputSequenceT: ControlInputSequence,
](
    mppi: Mppi[StateT, ControlInputSequenceT],
    model: DynamicalModel[StateT, StateBatch, ControlInputSequenceT, ControlInputBatch],
    current_state: StateT,
    nominal_input: ControlInputSequenceT,
    temperature: float,
    max_iterations: int,
    convergence_threshold: float,
) -> None:
    for _ in range(max_iterations):
        control = mppi.step(
            temperature=temperature,
            nominal_input=nominal_input,
            initial_state=current_state,
        )
        nominal_input = control.nominal

        if (
            np.linalg.norm(current_state := model.step(control.optimal, current_state))
            < convergence_threshold
        ):
            break

    assert (final_distance := np.linalg.norm(current_state)) < convergence_threshold, (
        f"MPPI did not converge to origin. "
        f"Final distance: {final_distance:.4f}, threshold: {convergence_threshold}"
    )


@mark.parametrize(
    ["mppi", "nominal_input", "initial_state"],
    [
        (
            mppi := create_mppi.numpy.base(
                model=stubs.DynamicalModel.returns(
                    rollouts := data.numpy.state_batch(
                        np.zeros((T := 2, D_x := 1, M := 3))
                    ),
                    when_control_inputs_are=(
                        sampled_inputs := data.numpy.control_input_batch(
                            array(
                                np.array(
                                    [
                                        sample_1 := [[1e-5], [1e-5]],
                                        sample_2 := [[2.0], [2.0]],
                                        sample_3 := [[3e10], [3e10]],
                                    ]
                                )
                                .transpose(1, 2, 0)
                                .tolist(),
                                shape=(T, D_u := 1, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.numpy.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.numpy.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.numpy.control_input_sequence(
                            np.zeros((T, D_u))
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
        ),
        (
            mppi := create_mppi.jax.base(
                model=stubs.DynamicalModel.returns(
                    rollouts := data.jax.state_batch(
                        np.zeros((T := 2, D_x := 1, M := 3))
                    ),
                    when_control_inputs_are=(
                        sampled_inputs := data.jax.control_input_batch(
                            array(
                                np.array(
                                    [
                                        sample_1 := [[1e-5], [1e-5]],
                                        sample_2 := [[2.0], [2.0]],
                                        sample_3 := [[3e10], [3e10]],
                                    ]
                                )
                                .transpose(1, 2, 0)
                                .tolist(),
                                shape=(T, D_u := 1, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.jax.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.jax.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.jax.control_input_sequence(
                            np.zeros((T, D_u))
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
        ),
    ],
)
def test_that_mppi_does_not_overflow_when_sample_cost_differences_are_very_large[
    StateT: State,
    ControlInputSequenceT: ControlInputSequence,
](
    mppi: Mppi[StateT, ControlInputSequenceT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
) -> None:
    control = mppi.step(
        temperature=1e-10,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.all(np.isfinite(control.optimal)), (
        f"Optimal control contains non-finite values: {control.optimal}"
    )


@mark.parametrize(
    ["mppi", "nominal_input", "initial_state"],
    [
        (
            mppi := create_mppi.numpy.base(
                model=stubs.DynamicalModel.returns(
                    rollouts := data.numpy.state_batch(
                        np.zeros((T := 2, D_x := 1, M := 3))
                    ),
                    when_control_inputs_are=(
                        sampled_inputs := data.numpy.control_input_batch(
                            array(
                                np.array(
                                    [
                                        sample_1 := [[1.0], [1.0]],
                                        sample_2 := [[1.0], [1.0]],
                                        sample_3 := [[1.0], [1.0]],
                                    ]
                                )
                                .transpose(1, 2, 0)
                                .tolist(),
                                shape=(T, D_u := 1, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.numpy.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.numpy.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.numpy.control_input_sequence(
                            np.zeros((T, D_u))
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
        ),
        (
            mppi := create_mppi.jax.base(
                model=stubs.DynamicalModel.returns(
                    rollouts := data.jax.state_batch(
                        np.zeros((T := 2, D_x := 1, M := 3))
                    ),
                    when_control_inputs_are=(
                        sampled_inputs := data.jax.control_input_batch(
                            array(
                                np.array(
                                    [
                                        sample_1 := [[1.0], [1.0]],
                                        sample_2 := [[1.0], [1.0]],
                                        sample_3 := [[1.0], [1.0]],
                                    ]
                                )
                                .transpose(1, 2, 0)
                                .tolist(),
                                shape=(T, D_u := 1, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.jax.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.jax.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.jax.control_input_sequence(
                            np.zeros((T, D_u))
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
        ),
    ],
)
def test_that_mppi_does_not_underflow_when_sample_cost_differences_are_very_small[
    StateT: State,
    ControlInputSequenceT: ControlInputSequence,
](
    mppi: Mppi[StateT, ControlInputSequenceT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
) -> None:
    control = mppi.step(
        temperature=1.0,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.all(np.isfinite(control.optimal)), (
        f"Optimal control contains non-finite values: {control.optimal}"
    )
    assert not np.allclose(control.optimal, 0.0), (
        f"Optimal control is unexpectedly zero: {control.optimal}"
    )


@mark.parametrize(
    ["mppi", "nominal_input", "initial_state", "expected_updated_input"],
    [
        (
            mppi := create_mppi.numpy.base(
                update_function=stubs.UpdateFunction.returns(
                    data.numpy.control_input_sequence(
                        array([[9.0, 9.0], [10.0, 10.0]], shape=(T := 2, D_u := 2))
                    ),
                    when_nominal_input_is=(
                        nominal_input := data.numpy.control_input_sequence(
                            array([[2.0, 3.0], [1.0, 2.0]], shape=(T, D_u))
                        )
                    ),
                    and_optimal_input_is=data.numpy.control_input_sequence(
                        array(sample := [[1.0, 2.0], [3.0, 4.0]], shape=(T, D_u))
                    ),
                ),
                model=stubs.DynamicalModel.returns(
                    rollouts := data.numpy.state_batch(np.zeros((T, D_x := 1, M := 1))),
                    when_control_inputs_are=(
                        sampled_inputs := data.numpy.control_input_batch(
                            array(
                                np.array([sample]).transpose(1, 2, 0).tolist(),
                                shape=(T, D_u, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.numpy.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.numpy.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs, when_nominal_input_is=nominal_input
                ),
            ),
            nominal_input,
            initial_state,
            # Should also be shifted
            expected_updated_input := data.numpy.control_input_sequence(
                array([[10.0, 10.0], [0.0, 0.0]], shape=(T, D_u))
            ),
        ),
        (
            mppi := create_mppi.jax.base(
                update_function=stubs.UpdateFunction.returns(
                    data.jax.control_input_sequence(
                        array([[10.0, 10.0], [1.0, 1.0]], shape=(T := 2, D_u := 2))
                    ),
                    when_nominal_input_is=(
                        nominal_input := data.jax.control_input_sequence(
                            array([[2.0, 4.0], [1.0, 1.0]], shape=(T, D_u))
                        )
                    ),
                    and_optimal_input_is=data.jax.control_input_sequence(
                        array(sample := [[2.0, 12.0], [21.0, 21.0]], shape=(T, D_u))
                    ),
                ),
                model=stubs.DynamicalModel.returns(
                    rollouts := data.jax.state_batch(np.zeros((T, D_x := 1, M := 1))),
                    when_control_inputs_are=(
                        sampled_inputs := data.jax.control_input_batch(
                            array(
                                np.array([sample]).transpose(1, 2, 0).tolist(),
                                shape=(T, D_u, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.jax.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.jax.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs, when_nominal_input_is=nominal_input
                ),
            ),
            nominal_input,
            initial_state,
            expected_updated_input := data.jax.control_input_sequence(
                array([[1.0, 1.0], [0.0, 0.0]], shape=(T, D_u))
            ),
        ),
    ],
)
def test_that_mppi_uses_update_function_to_update_nominal_input[
    StateT: State,
    ControlInputSequenceT: ControlInputSequence,
](
    mppi: Mppi[StateT, ControlInputSequenceT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    expected_updated_input: ControlInputSequenceT,
) -> None:
    control = mppi.step(
        temperature=1.0,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.allclose(control.nominal, expected_updated_input), (
        f"Expected nominal input to be updated to {expected_updated_input}, "
        f"but got {control.nominal} instead."
    )


T = clear_type
D_u = clear_type
padding_size = clear_type
sampled_inputs = clear_type


@mark.parametrize(
    ["mppi", "nominal_input", "initial_state", "expected_nominal_control"],
    [
        (
            mppi := create_mppi.numpy.base(
                planning_interval=(padding_size := 2),
                update_function=update.numpy.no_update(),
                padding_function=stubs.PaddingFunction.returns(  # type: ignore
                    data.numpy.control_input_sequence(
                        array(
                            [[12.0, 16.0], [17.0, 18.0]],
                            shape=(padding_size, D_u := 2),
                        )
                    ),
                    when_nominal_input_is=(
                        nominal_input := data.numpy.control_input_sequence(
                            array(
                                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                                shape=(T := 4, D_u),
                            )
                        )
                    ),
                    and_padding_size_is=padding_size,
                ),
                model=stubs.DynamicalModel.returns(
                    rollouts := data.numpy.state_batch(np.zeros((T, D_x := 1, M := 3))),
                    when_control_inputs_are=(
                        sampled_inputs := data.numpy.control_input_batch(
                            np.random.uniform(low=-1.0, high=1.0, size=(T, D_u, M))  # type: ignore
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.numpy.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.numpy.energy(),
                sampler=stubs.Sampler.returns(  # type: ignore
                    sampled_inputs, when_nominal_input_is=nominal_input
                ),
            ),
            nominal_input,
            initial_state,
            expected_nominal_control := data.numpy.control_input_sequence(
                array(
                    [[5.0, 6.0], [7.0, 8.0], [12.0, 16.0], [17.0, 18.0]], shape=(T, D_u)
                )
            ),
        ),
        (
            mppi := create_mppi.jax.base(
                planning_interval=(padding_size := 3),
                update_function=update.jax.no_update(),
                padding_function=stubs.PaddingFunction.returns(  # type: ignore
                    data.jax.control_input_sequence(
                        array(
                            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0]],
                            shape=(padding_size, D_u := 2),
                        )
                    ),
                    when_nominal_input_is=(
                        nominal_input := data.jax.control_input_sequence(
                            array(
                                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                                shape=(T := 4, D_u),
                            )
                        )
                    ),
                    and_padding_size_is=padding_size,
                ),
                model=stubs.DynamicalModel.returns(
                    rollouts := data.jax.state_batch(np.zeros((T, D_x := 1, M := 3))),
                    when_control_inputs_are=(
                        sampled_inputs_jax := data.jax.control_input_batch(
                            np.random.uniform(low=-1.0, high=1.0, size=(T, D_u, M))  # type: ignore
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.jax.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.jax.energy(),
                sampler=stubs.Sampler.returns(  # type: ignore
                    sampled_inputs_jax, when_nominal_input_is=nominal_input
                ),
            ),
            nominal_input,
            initial_state,
            expected_nominal_control := data.jax.control_input_sequence(
                array(
                    [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [13.0, 14.0]],
                    shape=(T, D_u),
                )
            ),
        ),
    ],
)
def test_that_mppi_uses_padding_function_to_pad_nominal_input[
    StateT: State,
    ControlInputSequenceT: ControlInputSequence,
](
    mppi: Mppi[StateT, ControlInputSequenceT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    expected_nominal_control: ControlInputSequenceT,
) -> None:
    control = mppi.step(
        temperature=1.0,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.allclose(control.nominal, expected_nominal_control), (
        f"Expected nominal input to be padded with custom values, "
        f"but got {control.nominal} instead of {expected_nominal_control}."
    )


T = clear_type
D_u = clear_type
D_x = clear_type
M = clear_type


@mark.parametrize(
    [
        "mppi",
        "nominal_input",
        "initial_state",
        "expected_optimal_control",
        "expected_nominal_control",
    ],
    [
        (
            mppi := create_mppi.numpy.base(
                update_function=update.numpy.use_optimal_control(),
                filter_function=stubs.FilterFunction.returns(
                    expected_optimal_control := data.numpy.control_input_sequence(
                        array(
                            [[10.0, 20.0], [30.0, 40.0]],
                            shape=(T := 2, D_u := 2),
                        )
                    ),
                    when_optimal_input_is=data.numpy.control_input_sequence(
                        array(sample := [[1.0, 2.0], [3.0, 4.0]], shape=(T, D_u))
                    ),
                ),
                model=stubs.DynamicalModel.returns(
                    rollouts := data.numpy.state_batch(np.zeros((T, D_x := 1, M := 1))),
                    when_control_inputs_are=(
                        sampled_inputs := data.numpy.control_input_batch(
                            array(
                                np.array([sample]).transpose(1, 2, 0).tolist(),
                                shape=(T, D_u, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.numpy.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.numpy.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs,
                    when_nominal_input_is=(
                        nominal_input := data.numpy.control_input_sequence(
                            array([[0.0, 0.0], [0.0, 0.0]], shape=(T, D_u))
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
            expected_optimal_control,
            # Since we are directly updating to the optimal control, nominal should be shifted optimal control.
            expected_nominal_control := data.numpy.control_input_sequence(
                array([[30.0, 40.0], [0.0, 0.0]], shape=(T, D_u))
            ),
        ),
        (
            mppi := create_mppi.jax.base(
                update_function=update.jax.use_optimal_control(),
                filter_function=stubs.FilterFunction.returns(
                    expected_optimal_control := data.jax.control_input_sequence(
                        array([[50.0, 60.0], [70.0, 80.0]], shape=(T := 2, D_u := 2))
                    ),
                    when_optimal_input_is=data.jax.control_input_sequence(
                        array(sample := [[5.0, 6.0], [7.0, 8.0]], shape=(T, D_u))
                    ),
                ),
                model=stubs.DynamicalModel.returns(  # type: ignore
                    rollouts := data.jax.state_batch(np.zeros((T, D_x := 1, M := 1))),
                    when_control_inputs_are=(
                        sampled_inputs_jax := data.jax.control_input_batch(
                            array(
                                np.array([sample]).transpose(1, 2, 0).tolist(),
                                shape=(T, D_u, M),
                            )
                        )
                    ),
                    and_initial_state_is=(
                        initial_state := data.jax.state(array([0.0], shape=(D_x,)))
                    ),
                ),
                cost_function=costs.jax.energy(),
                sampler=stubs.Sampler.returns(
                    sampled_inputs_jax,
                    when_nominal_input_is=(
                        nominal_input := data.jax.control_input_sequence(
                            array([[0.0, 0.0], [0.0, 0.0]], shape=(T, D_u))
                        )
                    ),
                ),
            ),
            nominal_input,
            initial_state,
            expected_optimal_control,
            expected_nominal_control := data.jax.control_input_sequence(
                array([[70.0, 80.0], [0.0, 0.0]], shape=(T, D_u))
            ),
        ),
    ],
)
def test_that_mppi_uses_filter_function_to_filter_optimal_control[
    StateT: State,
    ControlInputSequenceT: ControlInputSequence,
](
    mppi: Mppi[StateT, ControlInputSequenceT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    expected_optimal_control: ControlInputSequenceT,
    expected_nominal_control: ControlInputSequenceT,
) -> None:
    control = mppi.step(
        temperature=1.0,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.allclose(control.optimal, expected_optimal_control), (
        f"Expected optimal control to be filtered, "
        f"but got {control.optimal} instead of {expected_optimal_control}."
    )
    assert np.allclose(control.nominal, expected_nominal_control), (
        f"Expected nominal control to be {expected_nominal_control}, "
        f"but got {control.nominal} instead."
    )
