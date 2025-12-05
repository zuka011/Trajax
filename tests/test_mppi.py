# ==========================================
# 2. Numerical Stability
# ==========================================
# [ ] test_should_not_overflow_when_sample_cost_differences_are_very_large
#     - Ensures log_sum_exp trick is working for huge costs.

# [ ] test_should_not_underflow_when_sample_cost_differences_are_very_small
#     - Ensures division by zero doesn't occur when all costs are nearly identical.

# [ ] test_should_return_non_zero_and_finite_trajectory
#     - Basic smoke test ensuring outputs are not NaN/Inf.

# ==========================================
# 4. Strategies & Interfaces
# ==========================================
# [ ] test_should_pass_evaluation_key_to_cost_function
#     - Verifies API contract with the Cost module (e.g. JAX PRNG keys).

# [ ] test_should_not_throw_errors_when_using_the_specified_sampling_strategy
#     - Verifies API contract with various Sampler implementations.

# [ ] test_should_use_specified_update_strategy_when_updating_initial_control
#     - Verifies logic for padding/decaying the control sequence

from trajax import (
    State,
    StateBatch,
    ControlInputBatch,
    Mppi,
    DynamicalModel,
    Sampler,
    Costs,
    CostFunction,
    ControlInputSequence,
    mppi as create_mppi,
    update,
    padding,
    samplers,
)

import numpy as np
import jax.random as jrandom
from numtypes import array

from tests.dsl import mppi as data, integrator, costs, stubs, clear_type
from pytest import mark


@mark.asyncio
@mark.parametrize(
    [
        "mppi",
        "cost_function",
        "temperature",
        "nominal_input",
        "initial_state",
        "sampled_inputs",
        "rollouts",
        "expected_optimal_control",
        "tolerance",
    ],
    [
        (
            mppi := create_mppi.numpy(),
            cost_function := costs.numpy.energy(),
            temperature := 0.1,  # Low temperature, optimal should be close to best.
            nominal_input := data.numpy.control_input_sequence(
                np.random.uniform(low=-1.0, high=1.0, size=(T := 4, D_u := 3))
            ),
            initial_state := data.numpy.state(array([1.0, 2.0], shape=(D_x := 2,))),
            sampled_inputs := data.numpy.control_input_batch(
                array(
                    np.array(
                        [
                            # Medium energy sample.
                            np.random.uniform(low=-6.0, high=-5.0, size=(T, D_u)),
                            # Spends the least energy, should be favored.
                            optimal_control := (
                                np.random.uniform(low=-1.0, high=1.0, size=(T, D_u))
                            ),
                            # High energy sample.
                            np.random.uniform(low=10.0, high=11.0, size=(T, D_u)),
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T, D_u, M := 3),
                )
            ),
            rollouts := data.numpy.state_batch(  # Doesn't matter for this test case.
                np.random.uniform(low=0.0, high=1.0, size=(T, D_x, M))
            ),
            expected_optimal_control := data.numpy.control_input_sequence(
                optimal_control
            ),
            tolerance := 1e-3,  # Allow some tolerance, since T not that small.
        ),
        (  # Analogous test case for JAX implementation.
            mppi := create_mppi.jax(),
            cost_function := costs.jax.energy(),
            temperature := 0.1,
            nominal_input := data.jax.control_input_sequence(
                np.random.uniform(low=-3.0, high=4.0, size=(T := 4, D_u := 3))
            ),
            initial_state := data.jax.state(array([1.0, 2.0], shape=(D_x := 2,))),
            sampled_inputs := data.jax.control_input_batch(
                array(
                    np.array(
                        [
                            # High energy sample.
                            np.random.uniform(low=12.0, high=15.0, size=(T, D_u)),
                            optimal_control := (
                                np.random.uniform(low=-2.0, high=2.0, size=(T, D_u))
                            ),
                            # Medium energy sample.
                            np.random.uniform(low=-10.0, high=-7.0, size=(T, D_u)),
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T, D_u, M := 3),
                )
            ),
            rollouts := data.jax.state_batch(
                np.random.uniform(low=2.0, high=3.0, size=(T, D_x, M := 3))
            ),
            expected_optimal_control := data.jax.control_input_sequence(
                optimal_control
            ),
            tolerance := 1e-3,
        ),
    ],
)
async def test_that_mppi_favors_samples_with_lower_costs[
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
    CostsT: Costs,
](
    mppi: Mppi[StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT, CostsT],
    cost_function: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    temperature: float,
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    sampled_inputs: ControlInputBatchT,
    rollouts: StateBatchT,
    expected_optimal_control: ControlInputSequenceT,
    tolerance: float,
) -> None:
    sampler = stubs.Sampler.returns(sampled_inputs, when_nominal_input_is=nominal_input)
    model = stubs.DynamicalModel.returns(
        rollouts,
        when_control_inputs_are=sampled_inputs,
        and_initial_state_is=initial_state,
    )

    control = await mppi.step(
        model=model,
        cost_function=cost_function,
        sampler=sampler,
        temperature=temperature,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.allclose(control.optimal, expected_optimal_control, atol=tolerance)


T = clear_type


@mark.asyncio
@mark.parametrize(
    [
        "mppi",
        "cost_function",
        "nominal_input",
        "initial_state",
        "sampled_inputs",
        "rollouts",
        "expected_nominal_control",
    ],
    [
        (
            mppi := create_mppi.numpy(
                planning_interval=3,
                update_function=update.numpy.no_update(),
                padding_function=padding.numpy.zero(),
            ),
            cost_function := costs.numpy.energy(),
            nominal_input := data.numpy.control_input_sequence(
                array(
                    [
                        [1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0],
                        [7.0, 8.0],
                        [9.0, 10.0],
                    ],
                    shape=(T := 5, D_u := 2),
                )
            ),
            initial_state := data.numpy.state(array([0.0], shape=(D_x := 1,))),
            sampled_inputs := data.numpy.control_input_batch(
                np.random.uniform(low=-1.0, high=1.0, size=(T, D_u, M := 5))  # type: ignore
            ),
            rollouts := data.numpy.state_batch(np.zeros((T, D_x, M))),
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
            mppi := create_mppi.jax(
                planning_interval=4,
                update_function=update.jax.no_update(),
                padding_function=padding.jax.zero(),
            ),
            cost_function := costs.jax.energy(),
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
                    shape=(T := 6, D_u := 2),
                )
            ),
            initial_state := data.jax.state(array([0.0], shape=(D_x := 1,))),
            sampled_inputs := data.jax.control_input_batch(
                np.random.uniform(low=-1.0, high=1.0, size=(T, D_u, M := 4))  # type: ignore
            ),
            rollouts := data.jax.state_batch(np.zeros((T, D_x, M))),
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
async def test_that_mppi_shifts_control_sequence_left_when_replanning[
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
    CostsT: Costs,
](
    mppi: Mppi[StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT, CostsT],
    cost_function: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    sampled_inputs: ControlInputBatchT,
    rollouts: StateBatchT,
    expected_nominal_control: ControlInputSequenceT,
) -> None:
    sampler = stubs.Sampler.returns(sampled_inputs, when_nominal_input_is=nominal_input)
    model = stubs.DynamicalModel.returns(
        rollouts,
        when_control_inputs_are=sampled_inputs,
        and_initial_state_is=initial_state,
    )

    control = await mppi.step(
        model=model,
        cost_function=cost_function,
        sampler=sampler,
        temperature=1.0,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.allclose(control.nominal, expected_nominal_control)


@mark.asyncio
@mark.parametrize(
    [
        "mppi",
        "cost_function",
        "nominal_input",
        "initial_state",
        "sampled_inputs",
        "rollouts",
        "expected_optimal_control",
    ],
    [
        (
            mppi := create_mppi.numpy(),
            cost_function := costs.numpy.energy(),
            nominal_input := data.numpy.control_input_sequence(
                array([[0.0, 0.0], [0.0, 0.0]], shape=(T := 2, D_u := 2))
            ),
            initial_state := data.numpy.state(array([0.0], shape=(D_x := 1,))),
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
                    shape=(T, D_u, M := 3),
                )
            ),
            rollouts := data.numpy.state_batch(np.zeros((T, D_x, M))),
            # Result should be close to average of all controls.
            expected_optimal_control := data.numpy.control_input_sequence(
                array(
                    np.mean([control_1, control_2, control_3], axis=0), shape=(T, D_u)
                )
            ),
        ),
        (
            mppi := create_mppi.jax(),
            cost_function := costs.jax.energy(),
            nominal_input := data.jax.control_input_sequence(
                array([[0.0, 0.0], [0.0, 0.0]], shape=(T := 2, D_u := 2))
            ),
            initial_state := data.jax.state(array([0.0], shape=(D_x := 1,))),
            sampled_inputs := data.jax.control_input_batch(
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
                    shape=(T, D_u, M := 3),
                )
            ),
            rollouts := data.jax.state_batch(np.zeros((T, D_x, M))),
            expected_optimal_control := data.jax.control_input_sequence(
                array(
                    np.mean([control_1, control_2, control_3], axis=0), shape=(T, D_u)
                )
            ),
        ),
    ],
)
async def test_that_mppi_uses_samples_with_higher_costs_when_temperature_is_high[
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
    CostsT: Costs,
](
    mppi: Mppi[StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT, CostsT],
    cost_function: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    sampled_inputs: ControlInputBatchT,
    rollouts: StateBatchT,
    expected_optimal_control: ControlInputSequenceT,
) -> None:
    sampler = stubs.Sampler.returns(sampled_inputs, when_nominal_input_is=nominal_input)
    model = stubs.DynamicalModel.returns(
        rollouts,
        when_control_inputs_are=sampled_inputs,
        and_initial_state_is=initial_state,
    )

    control = await mppi.step(
        model=model,
        cost_function=cost_function,
        sampler=sampler,
        temperature=3e3,  # High temperature, should use all samples equally.
        nominal_input=nominal_input,
        initial_state=initial_state,
    )

    assert np.allclose(control.optimal, expected_optimal_control, rtol=0.1)


@mark.asyncio
@mark.parametrize(
    [
        "mppi",
        "cost_function",
        "nominal_input",
        "initial_state",
        "sampled_inputs",
        "rollouts",
    ],
    [
        (
            # If the weights sum to 1, then the optimal control should be a convex combination
            # of the samples, meaning it must be within the convex hull of the samples.
            mppi := create_mppi.numpy(),
            cost_function := costs.numpy.energy(),
            nominal_input := data.numpy.control_input_sequence(
                array([[0.0], [0.0]], shape=(T := 2, D_u := 1))
            ),
            initial_state := data.numpy.state(array([0.0], shape=(D_x := 1,))),
            sampled_inputs := data.numpy.control_input_batch(
                array(
                    np.array(
                        [[[3.0], [3.0]], [[-3.0], [-3.0]]],
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T, D_u, M := 2),
                )
            ),
            rollouts := data.numpy.state_batch(np.zeros((T, D_x, M))),
        ),
        (
            mppi := create_mppi.jax(),
            cost_function := costs.jax.energy(),
            nominal_input := data.jax.control_input_sequence(
                array([[0.0], [0.0]], shape=(T := 2, D_u := 1))
            ),
            initial_state := data.jax.state(array([0.0], shape=(D_x := 1,))),
            sampled_inputs := data.jax.control_input_batch(
                array(
                    np.array(
                        [[[3.0], [3.0]], [[-3.0], [-3.0]]],
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T, D_u, M := 2),
                )
            ),
            rollouts := data.jax.state_batch(np.zeros((T, D_x, M))),
        ),
    ],
)
async def test_that_mppi_optimal_control_is_convex_combination_of_samples[
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
    CostsT: Costs,
](
    mppi: Mppi[StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT, CostsT],
    cost_function: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    nominal_input: ControlInputSequenceT,
    initial_state: StateT,
    sampled_inputs: ControlInputBatchT,
    rollouts: StateBatchT,
) -> None:
    sampler = stubs.Sampler.returns(sampled_inputs, when_nominal_input_is=nominal_input)
    model = stubs.DynamicalModel.returns(
        rollouts,
        when_control_inputs_are=sampled_inputs,
        and_initial_state_is=initial_state,
    )

    control = await mppi.step(
        model=model,
        cost_function=cost_function,
        sampler=sampler,
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


@mark.asyncio
@mark.parametrize(
    [
        "mppi",
        "cost_function",
        "model",
        "sampler",
        "current_state",
        "nominal_input",
        "temperature",
        "max_iterations",
        "convergence_threshold",
    ],
    [
        (
            mppi := create_mppi.numpy(),
            cost_function := costs.numpy.quadratic_distance_to_origin(),
            model := integrator.numpy(time_step_size=0.1),
            sampler := samplers.sampler.numpy(
                standard_deviation=5.0,
                rollout_count=200,
                to_batch=data.numpy.control_input_batch,
                seed=42,
            ),
            current_state := data.numpy.state(array([5.0, 5.0], shape=(D_x := 2,))),
            nominal_input := data.numpy.control_input_sequence(
                np.zeros((T := 10, D_u := 2))
            ),
            temperature := 1.0,
            max_iterations := 100,
            convergence_threshold := 0.25,
        ),
        (
            mppi := create_mppi.jax(),
            cost_function := costs.jax.quadratic_distance_to_origin(),
            model := integrator.jax(time_step_size=0.1),
            sampler := samplers.sampler.jax(
                standard_deviation=5.0,
                rollout_count=200,
                to_batch=data.jax.control_input_batch,
                key=jrandom.PRNGKey(42),
            ),
            current_state := data.jax.state(array([5.0, 5.0], shape=(D_x := 2,))),
            nominal_input := data.jax.control_input_sequence(
                np.zeros((T := 10, D_u := 2))
            ),
            temperature := 1.0,
            max_iterations := 100,
            convergence_threshold := 0.25,
        ),
    ],
)
async def test_that_mppi_converges_to_target_state[
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
    CostsT: Costs,
](
    mppi: Mppi[StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT, CostsT],
    cost_function: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    model: DynamicalModel[
        ControlInputSequenceT, ControlInputBatchT, StateT, StateBatchT
    ],
    sampler: Sampler[ControlInputSequenceT, ControlInputBatchT],
    current_state: StateT,
    nominal_input: ControlInputSequenceT,
    temperature: float,
    max_iterations: int,
    convergence_threshold: float,
) -> None:
    for _ in range(max_iterations):
        control = await mppi.step(
            model=model,
            cost_function=cost_function,
            sampler=sampler,
            temperature=temperature,
            nominal_input=nominal_input,
            initial_state=current_state,
        )
        nominal_input = control.nominal

        if (
            np.linalg.norm(
                current_state := await model.step(control.optimal, current_state)
            )
            < convergence_threshold
        ):
            break

    assert (final_distance := np.linalg.norm(current_state)) < convergence_threshold, (
        f"MPPI did not converge to origin. "
        f"Final distance: {final_distance:.4f}, threshold: {convergence_threshold}"
    )
