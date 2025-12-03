# ==========================================
# 1. Core Algorithm Logic
# ==========================================
# [X] test_should_use_the_best_sample_for_the_next_control_signal
#     - Verifies that better samples (lower cost) dominate the resulting control (Weighted Averaging).

# [ ] test_should_shift_initial_control_signal_when_replanning
#     - Verifies that the control sequence shifts left (receding horizon) and is padded correctly.

# [ ] test_should_update_initial_control_signal_for_next_iteration
#     - Verifies that the new "initial guess" is improved based on the previous iteration (Warm Start).

# [ ] test_should_converge_to_the_target_state
#     - End-to-end sanity check ensuring the car actually gets there.

# [ ] test_zero_temperature_behavior (Hard Max)
#     - Verify that as temperature -> 0, the algorithm returns exactly the single best sampled trajectory (no averaging).

# [ ] test_weights_sum_to_one
#     - Unit test checking that the internal importance sampling weights always sum to 1.0, even with wildly different costs.

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
# 3. Control Features & Constraints
# ==========================================
# [ ] test_control_constraints_are_respected (Clamping) (Debatable if this is needed)
#     - Configure bounds (e.g., min=-1.0, max=1.0).
#     - Force the cost function to prefer a value of 100.0.
#     - Assert that the returned u_star is exactly 1.0.

# [ ] test_noise_is_added_correctly
#     - Verify that u_sample = u_nominal + noise logic holds true.
#     - Ensure noise isn't being double-counted or ignored during integration.

# ==========================================
# 4. Strategies & Interfaces (Mock Interaction)
# ==========================================
# [ ] test_should_pass_evaluation_key_to_cost_function
#     - Verifies API contract with the Cost module (e.g. JAX PRNG keys).

# [ ] test_should_not_throw_errors_when_using_the_specified_sampling_strategy
#     - Verifies API contract with various Sampler implementations.

# [ ] test_should_use_specified_update_strategy_when_updating_initial_control
#     - Verifies logic for padding/decaying the control sequence (e.g., SequentialMeanUpdate).
from dataclasses import dataclass

from trajax import (
    State,
    StateBatch,
    ControlInputBatch,
    Mppi,
    NumPyMppi,
    JaxMppi,
    Costs,
    CostFunction,
    ControlInputSequence,
)

import numpy as np
from numtypes import array

from tests.dsl import mppi as data, costs
from pytest import mark


@dataclass(frozen=True)
class StubSampler[SequenceT: ControlInputSequence, BatchT: ControlInputBatch]:
    samples: BatchT
    expected_nominal_input: SequenceT

    @staticmethod
    def returns[S: ControlInputSequence, B: ControlInputBatch](
        sampled_inputs: B, *, when_nominal_input_is: S
    ) -> "StubSampler[S, B]":
        return StubSampler(
            samples=sampled_inputs, expected_nominal_input=when_nominal_input_is
        )

    def sample(self, *, around: SequenceT) -> BatchT:
        assert np.array_equal(self.expected_nominal_input, around), (
            f"Sampler received an unexpected nominal input. "
            f"Expected: {self.expected_nominal_input}, Got: {around}"
        )
        return self.samples


@dataclass(frozen=True)
class StubDynamicalModel[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
]:
    rollouts: StateBatchT
    expected_control_inputs: ControlInputBatchT
    expected_initial_state: StateT

    @staticmethod
    def returns[SB: StateBatch, CIB: ControlInputBatch, S: State](
        rollouts: SB, *, when_control_inputs_are: CIB, and_initial_state_is: S
    ) -> "StubDynamicalModel[CIB, S, SB]":
        return StubDynamicalModel(
            rollouts=rollouts,
            expected_control_inputs=when_control_inputs_are,
            expected_initial_state=and_initial_state_is,
        )

    async def simulate(
        self, inputs: ControlInputBatchT, initial_state: StateT
    ) -> StateBatchT:
        assert np.array_equal(self.expected_control_inputs, inputs), (
            f"Dynamical model received unexpected control inputs. "
            f"Expected: {self.expected_control_inputs}, Got: {inputs}"
        )
        assert np.array_equal(self.expected_initial_state, initial_state), (
            f"Dynamical model received an unexpected initial state. "
            f"Expected: {self.expected_initial_state}, Got: {initial_state}"
        )
        return self.rollouts


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
            mppi := NumPyMppi.create(),
            cost_function := costs.numpy.energy(),
            temperature := 0.1,  # Low temperature, optimal should be close to best.
            nominal_input := data.numpy.control_input_sequence(
                array(
                    [
                        [0.1, 0.2, 0.3],
                        [0.3, 0.2, 0.1],
                        [0.0, 0.0, 0.0],
                        [0.2, 0.2, 0.2],
                    ],
                    shape=(T := 4, D_u := 3),
                )
            ),
            initial_state := data.numpy.state(array([1.0, 2.0], shape=(D_x := 2,))),
            sampled_inputs := data.numpy.control_input_batch(
                array(
                    np.array(
                        [
                            [
                                [5.0, 2.5, -0.8],
                                [0.3, -0.2, 1.5],
                                [0.0, -3.5, 4.4],
                                [-0.2, 0.1, 0.3],
                            ],
                            # Spends the least energy, should be favored.
                            optimal_control := array(
                                [
                                    [0.3, 0.1, 0.2],
                                    [0.1, 0.3, 0.2],
                                    [0.0, 0.0, 0.0],
                                    [0.2, 0.2, 0.2],
                                ],
                                shape=(T, D_u),
                            ),
                            [
                                [10.2, 10.4, -10.1],
                                [-10.2, 10.1, 10.3],
                                [10.0, -10.0, 10.0],
                                [10.1, 10.2, -10.4],
                            ],
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T, D_u, M := 3),
                )
            ),
            # Doesn't matter for this test case.
            rollouts := data.numpy.state_batch(
                np.random.uniform(low=0.0, high=1.0, size=(T, D_x, M))
            ),
            expected_optimal_control := data.numpy.control_input_sequence(
                optimal_control
            ),
            tolerance := 0.1,  # Allow some tolerance, since T not that small.
        ),
        (  # Analogous test case for JAX implementation.
            mppi := JaxMppi.create(),
            cost_function := costs.jax.energy(),
            temperature := 0.1,
            nominal_input := data.jax.control_input_sequence(
                array(
                    [
                        [0.1, 0.2, 0.3],
                        [0.3, 0.2, 0.1],
                        [0.0, 0.0, 0.0],
                        [0.2, 0.2, 0.2],
                    ],
                    shape=(T := 4, D_u := 3),
                )
            ),
            initial_state := data.jax.state(array([1.0, 2.0], shape=(D_x := 2,))),
            sampled_inputs := data.jax.control_input_batch(
                array(
                    np.array(
                        [
                            [
                                [5.0, 2.5, -0.8],
                                [0.3, -0.2, 1.5],
                                [0.0, -3.5, 4.4],
                                [-0.2, 0.1, 0.3],
                            ],
                            optimal_control := array(
                                [
                                    [0.3, 0.1, 0.2],
                                    [0.1, 0.3, 0.2],
                                    [0.0, 0.0, 0.0],
                                    [0.2, 0.2, 0.2],
                                ],
                                shape=(T, D_u),
                            ),
                            [
                                [10.2, 10.4, -10.1],
                                [-10.2, 10.1, 10.3],
                                [10.0, -10.0, 10.0],
                                [10.1, 10.2, -10.4],
                            ],
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
            tolerance := 0.1,
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
    sampler = StubSampler.returns(sampled_inputs, when_nominal_input_is=nominal_input)
    model = StubDynamicalModel.returns(
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
