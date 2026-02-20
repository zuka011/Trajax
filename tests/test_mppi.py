from typing import Sequence

from faran import (
    State,
    StateSequence,
    StateBatch,
    ControlInputBatch,
    Mppi,
    DynamicalModel,
    ControlInputSequence,
    mppi as create_mppi,
    model as create_model,
    update,
    padding,
    sampler,
)

import numpy as np
from numtypes import array

from tests.dsl import mppi as data, costs, stubs
from pytest import mark


class test_that_mppi_favors_samples_with_lower_costs:
    @staticmethod
    def cases(create_mppi, data, costs) -> Sequence[tuple]:
        return [
            (
                mppi := create_mppi.base(
                    model=stubs.DynamicalModel.returns(
                        rollouts=data.state_batch(  # Doesn't matter for this test case.
                            np.random.uniform(
                                low=0.0, high=1.0, size=(T := 4, D_x := 2, M := 3)
                            )
                        ),
                        when_control_inputs_are=(
                            sampled_inputs := data.control_input_batch(
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
                            initial_state := data.state(array([1.0, 2.0], shape=(D_x,)))
                        ),
                    ),
                    cost_function=costs.energy(),
                    sampler=stubs.Sampler.returns(
                        sampled_inputs,
                        when_nominal_input_is=(
                            nominal_input := data.control_input_sequence(
                                np.random.uniform(low=-1.0, high=1.0, size=(T, D_u))
                            )
                        ),
                    ),
                ),
                temperature := 0.1,  # Low temperature, optimal should be close to best.
                nominal_input,
                initial_state,
                expected_optimal_control := data.control_input_sequence(
                    optimal_control
                ),
                # Allow some tolerance, since temp. is not that small.
                tolerance := 1e-3,
            ),
        ]

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
            *cases(create_mppi=create_mppi.numpy, data=data.numpy, costs=costs.numpy),
            *cases(create_mppi=create_mppi.jax, data=data.jax, costs=costs.jax),
        ],
    )
    def test[StateT: State, ControlInputSequenceT: ControlInputSequence](
        self,
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


class test_that_mppi_shifts_control_sequence_left_when_replanning:
    @staticmethod
    def cases(create_mppi, update, padding, data, costs) -> Sequence[tuple]:
        return [
            (
                mppi := create_mppi.base(
                    planning_interval=3,
                    update_function=update.none(),
                    padding_function=padding.zero(),
                    model=stubs.DynamicalModel.returns(
                        rollouts := data.state_batch(
                            np.zeros((T := 5, D_x := 1, M := 5))
                        ),
                        when_control_inputs_are=(
                            sampled_inputs := data.control_input_batch(
                                np.random.uniform(
                                    low=-1.0, high=1.0, size=(T, D_u := 2, M)
                                )
                            )
                        ),
                        and_initial_state_is=(
                            initial_state := data.state(array([0.0], shape=(D_x,)))
                        ),
                    ),
                    cost_function=costs.energy(),
                    sampler=stubs.Sampler.returns(
                        sampled_inputs,
                        when_nominal_input_is=(
                            nominal_input := data.control_input_sequence(
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
                expected_nominal_control := data.control_input_sequence(
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
        ]

    @mark.parametrize(
        ["mppi", "nominal_input", "initial_state", "expected_nominal_control"],
        [
            *cases(
                create_mppi=create_mppi.numpy,
                update=update.numpy,
                padding=padding.numpy,
                data=data.numpy,
                costs=costs.numpy,
            ),
            *cases(
                create_mppi=create_mppi.jax,
                update=update.jax,
                padding=padding.jax,
                data=data.jax,
                costs=costs.jax,
            ),
        ],
    )
    def test[StateT: State, ControlInputSequenceT: ControlInputSequence](
        self,
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


class test_that_mppi_uses_samples_with_higher_costs_when_temperature_is_high:
    @staticmethod
    def cases(create_mppi, data, costs) -> Sequence[tuple]:
        return [
            (
                mppi := create_mppi.base(
                    model=stubs.DynamicalModel.returns(
                        rollouts := data.state_batch(
                            np.zeros((T := 2, D_x := 1, M := 3))
                        ),
                        when_control_inputs_are=(
                            sampled_inputs := data.control_input_batch(
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
                            initial_state := data.state(array([0.0], shape=(D_x,)))
                        ),
                    ),
                    cost_function=costs.energy(),
                    sampler=stubs.Sampler.returns(
                        sampled_inputs,
                        when_nominal_input_is=(
                            nominal_input := data.control_input_sequence(
                                array([[0.0, 0.0], [0.0, 0.0]], shape=(T, D_u))
                            )
                        ),
                    ),
                ),
                nominal_input,
                initial_state,
                # Result should be close to average of all controls.
                expected_optimal_control := data.control_input_sequence(
                    array(
                        np.mean([control_1, control_2, control_3], axis=0),
                        shape=(T, D_u),
                    )
                ),
            )
        ]

    @mark.parametrize(
        ["mppi", "nominal_input", "initial_state", "expected_optimal_control"],
        [
            *cases(create_mppi=create_mppi.numpy, data=data.numpy, costs=costs.numpy),
            *cases(create_mppi=create_mppi.jax, data=data.jax, costs=costs.jax),
        ],
    )
    def test[StateT: State, ControlInputSequenceT: ControlInputSequence](
        self,
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


class test_that_mppi_optimal_control_is_convex_combination_of_samples:
    @staticmethod
    def cases(create_mppi, data, costs) -> Sequence[tuple]:
        return [
            (
                # If the weights sum to 1, then the optimal control should be a convex combination
                # of the samples, meaning it must be within the convex hull of the samples.
                mppi := create_mppi.base(
                    model=stubs.DynamicalModel.returns(
                        rollouts := data.state_batch(
                            np.zeros((T := 2, D_x := 1, M := 2))
                        ),
                        when_control_inputs_are=(
                            sampled_inputs := data.control_input_batch(
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
                            initial_state := data.state(array([0.0], shape=(D_x,)))
                        ),
                    ),
                    cost_function=costs.energy(),
                    sampler=stubs.Sampler.returns(
                        sampled_inputs,
                        when_nominal_input_is=(
                            nominal_input := data.control_input_sequence(
                                array([[0.0], [0.0]], shape=(T, D_u))
                            )
                        ),
                    ),
                ),
                nominal_input,
                initial_state,
                sampled_inputs,
            )
        ]

    @mark.parametrize(
        ["mppi", "nominal_input", "initial_state", "sampled_inputs"],
        [
            *cases(create_mppi=create_mppi.numpy, data=data.numpy, costs=costs.numpy),
            *cases(create_mppi=create_mppi.jax, data=data.jax, costs=costs.jax),
        ],
    )
    def test[StateT: State, ControlInputSequenceT: ControlInputSequence](
        self,
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


class test_that_mppi_converges_to_target_state:
    @staticmethod
    def cases(create_mppi, create_model, data, costs, sampler) -> Sequence[tuple]:
        return [
            (
                mppi := create_mppi.base(
                    model=(
                        model := create_model.integrator.dynamical(time_step_size=0.1)
                    ),
                    cost_function=costs.quadratic_distance_to_origin(),
                    sampler=sampler.gaussian(
                        standard_deviation=array([5.0, 5.0], shape=(D_u := 2,)),
                        rollout_count=200,
                        to_batch=data.control_input_batch,
                        seed=42,
                    ),
                ),
                model,
                current_state := data.state(array([224.0, 52.0], shape=(D_x := 2,))),
                nominal_input := data.control_input_sequence(np.zeros((T := 10, D_u))),
                temperature := 1.0,
                max_iterations := 100,
                convergence_threshold := 0.1,
            ),
        ]

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
            *cases(
                create_mppi=create_mppi.numpy,
                create_model=create_model.numpy,
                data=data.numpy,
                costs=costs.numpy,
                sampler=sampler.numpy,
            ),
            *cases(
                create_mppi=create_mppi.jax,
                create_model=create_model.jax,
                data=data.jax,
                costs=costs.jax,
                sampler=sampler.jax,
            ),
        ],
    )
    def test[StateT: State, ControlInputSequenceT: ControlInputSequence](
        self,
        mppi: Mppi[StateT, ControlInputSequenceT],
        model: DynamicalModel[
            StateT, StateSequence, StateBatch, ControlInputSequenceT, ControlInputBatch
        ],
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
                np.linalg.norm(
                    current_state := model.step(control.optimal, current_state)
                )
                < convergence_threshold
            ):
                break

        assert (
            final_distance := np.linalg.norm(current_state)
        ) < convergence_threshold, (
            f"MPPI did not converge to origin. "
            f"Final distance: {final_distance:.4f}, threshold: {convergence_threshold}"
        )


class test_that_mppi_does_not_overflow_when_sample_cost_differences_are_very_large:
    @staticmethod
    def cases(create_mppi, data, costs) -> Sequence[tuple]:
        return [
            (
                mppi := create_mppi.base(
                    model=stubs.DynamicalModel.returns(
                        rollouts := data.state_batch(
                            np.zeros((T := 2, D_x := 1, M := 3))
                        ),
                        when_control_inputs_are=(
                            sampled_inputs := data.control_input_batch(
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
                            initial_state := data.state(array([0.0], shape=(D_x,)))
                        ),
                    ),
                    cost_function=costs.energy(),
                    sampler=stubs.Sampler.returns(
                        sampled_inputs,
                        when_nominal_input_is=(
                            nominal_input := data.control_input_sequence(
                                np.zeros((T, D_u))
                            )
                        ),
                    ),
                ),
                nominal_input,
                initial_state,
            ),
        ]

    @mark.parametrize(
        ["mppi", "nominal_input", "initial_state"],
        [
            *cases(create_mppi=create_mppi.numpy, data=data.numpy, costs=costs.numpy),
            *cases(create_mppi=create_mppi.jax, data=data.jax, costs=costs.jax),
        ],
    )
    def test[StateT: State, ControlInputSequenceT: ControlInputSequence](
        self,
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


class test_that_mppi_does_not_underflow_when_sample_cost_differences_are_very_small:
    @staticmethod
    def cases(create_mppi, data, costs) -> Sequence[tuple]:
        return [
            (
                mppi := create_mppi.base(
                    model=stubs.DynamicalModel.returns(
                        rollouts := data.state_batch(
                            np.zeros((T := 2, D_x := 1, M := 3))
                        ),
                        when_control_inputs_are=(
                            sampled_inputs := data.control_input_batch(
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
                            initial_state := data.state(array([0.0], shape=(D_x,)))
                        ),
                    ),
                    cost_function=costs.energy(),
                    sampler=stubs.Sampler.returns(
                        sampled_inputs,
                        when_nominal_input_is=(
                            nominal_input := data.control_input_sequence(
                                np.zeros((T, D_u))
                            )
                        ),
                    ),
                ),
                nominal_input,
                initial_state,
            ),
        ]

    @mark.parametrize(
        ["mppi", "nominal_input", "initial_state"],
        [
            *cases(create_mppi=create_mppi.numpy, data=data.numpy, costs=costs.numpy),
            *cases(create_mppi=create_mppi.jax, data=data.jax, costs=costs.jax),
        ],
    )
    def test[StateT: State, ControlInputSequenceT: ControlInputSequence](
        self,
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


class test_that_mppi_uses_update_function_to_update_nominal_input:
    @staticmethod
    def cases(create_mppi, data, costs) -> Sequence[tuple]:
        return [
            (
                mppi := create_mppi.base(
                    update_function=stubs.UpdateFunction.returns(
                        data.control_input_sequence(
                            array([[9.0, 9.0], [10.0, 10.0]], shape=(T := 2, D_u := 2))
                        ),
                        when_nominal_input_is=(
                            nominal_input := data.control_input_sequence(
                                array([[2.0, 3.0], [1.0, 2.0]], shape=(T, D_u))
                            )
                        ),
                        and_optimal_input_is=data.control_input_sequence(
                            array(sample := [[1.0, 2.0], [3.0, 4.0]], shape=(T, D_u))
                        ),
                    ),
                    model=stubs.DynamicalModel.returns(
                        rollouts := data.state_batch(np.zeros((T, D_x := 1, M := 1))),
                        when_control_inputs_are=(
                            sampled_inputs := data.control_input_batch(
                                array(
                                    np.array([sample]).transpose(1, 2, 0).tolist(),
                                    shape=(T, D_u, M),
                                )
                            )
                        ),
                        and_initial_state_is=(
                            initial_state := data.state(array([0.0], shape=(D_x,)))
                        ),
                    ),
                    cost_function=costs.energy(),
                    sampler=stubs.Sampler.returns(
                        sampled_inputs, when_nominal_input_is=nominal_input
                    ),
                ),
                nominal_input,
                initial_state,
                # Should also be shifted
                expected_updated_input := data.control_input_sequence(
                    array([[10.0, 10.0], [0.0, 0.0]], shape=(T, D_u))
                ),
            ),
        ]

    @mark.parametrize(
        ["mppi", "nominal_input", "initial_state", "expected_updated_input"],
        [
            *cases(create_mppi=create_mppi.numpy, data=data.numpy, costs=costs.numpy),
            *cases(create_mppi=create_mppi.jax, data=data.jax, costs=costs.jax),
        ],
    )
    def test[StateT: State, ControlInputSequenceT: ControlInputSequence](
        self,
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


class test_that_mppi_uses_padding_function_to_pad_nominal_input:
    @staticmethod
    def cases(create_mppi, update, data, costs) -> Sequence[tuple]:
        return [
            (
                mppi := create_mppi.base(
                    planning_interval=(padding_size := 2),
                    update_function=update.none(),
                    padding_function=stubs.PaddingFunction.returns(
                        data.control_input_sequence(
                            array(
                                [[12.0, 16.0], [17.0, 18.0]],
                                shape=(padding_size, D_u := 2),
                            )
                        ),
                        when_nominal_input_is=(
                            nominal_input := data.control_input_sequence(
                                array(
                                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                                    shape=(T := 4, D_u),
                                )
                            )
                        ),
                        and_padding_size_is=padding_size,
                    ),
                    model=stubs.DynamicalModel.returns(
                        rollouts := data.state_batch(np.zeros((T, D_x := 1, M := 3))),
                        when_control_inputs_are=(
                            sampled_inputs := data.control_input_batch(
                                np.random.uniform(low=-1.0, high=1.0, size=(T, D_u, M))  # type: ignore
                            )
                        ),
                        and_initial_state_is=(
                            initial_state := data.state(array([0.0], shape=(D_x,)))
                        ),
                    ),
                    cost_function=costs.energy(),
                    sampler=stubs.Sampler.returns(  # type: ignore
                        sampled_inputs, when_nominal_input_is=nominal_input
                    ),
                ),
                nominal_input,
                initial_state,
                expected_nominal_control := data.control_input_sequence(
                    array(
                        [[5.0, 6.0], [7.0, 8.0], [12.0, 16.0], [17.0, 18.0]],
                        shape=(T, D_u),
                    )
                ),
            ),
        ]

    @mark.parametrize(
        ["mppi", "nominal_input", "initial_state", "expected_nominal_control"],
        [
            *cases(
                create_mppi=create_mppi.numpy,
                update=update.numpy,
                data=data.numpy,
                costs=costs.numpy,
            ),
            *cases(
                create_mppi=create_mppi.jax,
                update=update.jax,
                data=data.jax,
                costs=costs.jax,
            ),
        ],
    )
    def test[StateT: State, ControlInputSequenceT: ControlInputSequence](
        self,
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


class test_that_mppi_uses_filter_function_to_filter_optimal_control:
    @staticmethod
    def cases(create_mppi, update, data, costs) -> Sequence[tuple]:
        return [
            (
                mppi := create_mppi.base(
                    update_function=update.use_optimal_control(),
                    filter_function=stubs.FilterFunction.returns(
                        expected_optimal_control := data.control_input_sequence(
                            array(
                                [[10.0, 20.0], [30.0, 40.0]],
                                shape=(T := 2, D_u := 2),
                            )
                        ),
                        when_optimal_input_is=data.control_input_sequence(
                            array(sample := [[1.0, 2.0], [3.0, 4.0]], shape=(T, D_u))
                        ),
                    ),
                    model=stubs.DynamicalModel.returns(
                        rollouts := data.state_batch(np.zeros((T, D_x := 1, M := 1))),
                        when_control_inputs_are=(
                            sampled_inputs := data.control_input_batch(
                                array(
                                    np.array([sample]).transpose(1, 2, 0).tolist(),
                                    shape=(T, D_u, M),
                                )
                            )
                        ),
                        and_initial_state_is=(
                            initial_state := data.state(array([0.0], shape=(D_x,)))
                        ),
                    ),
                    cost_function=costs.energy(),
                    sampler=stubs.Sampler.returns(
                        sampled_inputs,
                        when_nominal_input_is=(
                            nominal_input := data.control_input_sequence(
                                array([[0.0, 0.0], [0.0, 0.0]], shape=(T, D_u))
                            )
                        ),
                    ),
                ),
                nominal_input,
                initial_state,
                expected_optimal_control,
                # Since we are directly updating to the optimal control, nominal should be shifted optimal control.
                expected_nominal_control := data.control_input_sequence(
                    array([[30.0, 40.0], [0.0, 0.0]], shape=(T, D_u))
                ),
            ),
        ]

    @mark.parametrize(
        [
            "mppi",
            "nominal_input",
            "initial_state",
            "expected_optimal_control",
            "expected_nominal_control",
        ],
        [
            *cases(
                create_mppi=create_mppi.numpy,
                update=update.numpy,
                data=data.numpy,
                costs=costs.numpy,
            ),
            *cases(
                create_mppi=create_mppi.jax,
                update=update.jax,
                data=data.jax,
                costs=costs.jax,
            ),
        ],
    )
    def test[StateT: State, ControlInputSequenceT: ControlInputSequence](
        self,
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
