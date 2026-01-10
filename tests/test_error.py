from typing import Sequence

from trajax import costs, trajectory, types, ContouringCost, LagCost, Error

from numtypes import array, Array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


type NumPyPathParameterExtractor = types.numpy.PathParameterExtractor[
    types.numpy.StateBatch
]
type JaxPathParameterExtractor = types.jax.PathParameterExtractor[types.jax.StateBatch]


def numpy_path_parameter_extractor(
    extractor: NumPyPathParameterExtractor,
) -> NumPyPathParameterExtractor:
    return extractor


def jax_path_parameter_extractor(
    extractor: JaxPathParameterExtractor,
) -> JaxPathParameterExtractor:
    return extractor


class test_that_contouring_cost_computes_correct_lateral_error:
    @staticmethod
    def cases(
        costs, trajectory, path_parameter_extractor, types, data
    ) -> Sequence[tuple]:
        return [
            (
                contouring_cost := costs.tracking.contouring(
                    reference=trajectory.waypoints(
                        points=array([[0.0, 0.0], [10.0, 10.0]], shape=(2, 2)),
                        path_length=(path_length := 10.0),
                    ),
                    path_parameter_extractor=path_parameter_extractor(
                        lambda states: types.path_parameters(states.array[:, 0, :])
                    ),
                    position_extractor=(
                        lambda states: types.positions(
                            x=states.array[:, 1, :], y=states.array[:, 2, :]
                        )
                    ),
                    weight=1.0,
                ),
                inputs := data.control_input_batch(
                    np.zeros((T := 1, D_u := 1, M := 1))
                ),
                states := data.state_batch(
                    array(
                        [[[phi := 5.0], [x := 7.5], [y := 7.5]]], shape=(T, D_x := 3, M)
                    )
                ),
                expected_error := 0.0,
            ),
            *[
                (
                    contouring_cost := costs.tracking.contouring(
                        reference=trajectory.waypoints(
                            points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)),
                            path_length=10.0,
                        ),
                        path_parameter_extractor=path_parameter_extractor(
                            lambda states: types.path_parameters(states.array[:, 0, :])
                        ),
                        position_extractor=(
                            lambda states: types.positions(
                                x=states.array[:, 1, :], y=states.array[:, 2, :]
                            )
                        ),
                        weight=1.0,
                    ),
                    inputs := data.control_input_batch(
                        np.zeros((T := 1, D_u := 1, M := 1))
                    ),
                    states := data.state_batch(
                        array(
                            [[[phi], [5.0], [lateral_deviation]]],
                            shape=(T, D_x := 3, M),
                        )
                    ),
                    # Left of path is negative error
                    expected_error := -lateral_deviation,
                )
                for lateral_deviation in (2.0, -0.5)
                for phi in (0.0, 5.0)
            ],
        ]

    @mark.parametrize(
        ["contouring_cost", "inputs", "states", "expected_error"],
        [
            *cases(
                costs=costs.numpy,
                trajectory=trajectory.numpy,
                path_parameter_extractor=numpy_path_parameter_extractor,
                types=types.numpy,
                data=data.numpy,
            ),
            *cases(
                costs=costs.jax,
                trajectory=trajectory.jax,
                path_parameter_extractor=jax_path_parameter_extractor,
                types=types.jax,
                data=data.jax,
            ),
        ],
    )
    def test[InputBatchT, StateBatchT](
        self,
        contouring_cost: ContouringCost[InputBatchT, StateBatchT, Error],
        inputs: InputBatchT,
        states: StateBatchT,
        expected_error: float,
    ) -> None:
        error = contouring_cost.error(inputs=inputs, states=states)
        assert np.allclose(error, expected_error, atol=1e-6)


class test_that_contouring_error_handles_multiple_timesteps_and_rollouts:
    @staticmethod
    def cases(
        costs, trajectory, path_parameter_extractor, types, data
    ) -> Sequence[tuple]:
        return [
            (
                contouring_cost := costs.tracking.contouring(
                    reference=trajectory.waypoints(
                        points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)),
                        path_length=10.0,
                    ),
                    path_parameter_extractor=path_parameter_extractor(
                        lambda states: types.path_parameters(states.array[:, 0, :])
                    ),
                    position_extractor=(
                        lambda states: types.positions(
                            x=states.array[:, 1, :], y=states.array[:, 2, :]
                        )
                    ),
                    weight=1.0,
                ),
                inputs := data.control_input_batch(
                    np.zeros((T := 2, D_u := 1, M := 3))
                ),
                states := data.state_batch(
                    array(
                        [
                            [[2.0, 5.0, 8.0], [2.0, 5.0, 8.0], [0.5, 1.0, -1.5]],
                            [[3.0, 6.0, 9.0], [3.0, 6.0, 9.0], [1.0, -2.0, 0.0]],
                        ],
                        shape=(T, D_x := 3, M),
                    )
                ),
                expected_errors := array(
                    [[0.5, 1.0, 1.5], [1.0, 2.0, 0.0]], shape=(T, M)
                ),
            ),
        ]

    @mark.parametrize(
        ["contouring_cost", "inputs", "states", "expected_errors"],
        [
            *cases(
                costs=costs.numpy,
                trajectory=trajectory.numpy,
                path_parameter_extractor=numpy_path_parameter_extractor,
                types=types.numpy,
                data=data.numpy,
            ),
            *cases(
                costs=costs.jax,
                trajectory=trajectory.jax,
                path_parameter_extractor=jax_path_parameter_extractor,
                types=types.jax,
                data=data.jax,
            ),
        ],
    )
    def test[InputBatchT, StateBatchT](
        self,
        contouring_cost: ContouringCost[InputBatchT, StateBatchT, Error],
        inputs: InputBatchT,
        states: StateBatchT,
        expected_errors: Array,
    ) -> None:
        error = contouring_cost.error(inputs=inputs, states=states)
        assert np.allclose(np.abs(error), expected_errors, atol=1e-10)


class test_that_lag_cost_computes_correct_longitudinal_error:
    @staticmethod
    def cases(
        costs, trajectory, path_parameter_extractor, types, data
    ) -> Sequence[tuple]:
        return [
            *[  # Ego progress along path exactly matches reference
                (
                    lag_cost := costs.tracking.lag(
                        reference=trajectory.waypoints(
                            points=array([[0.0, 0.0], [10.0, 10.0]], shape=(2, 2)),
                            path_length=(path_length := 10.0),
                        ),
                        path_parameter_extractor=path_parameter_extractor(
                            lambda states: types.path_parameters(states.array[:, 0, :])
                        ),
                        position_extractor=(
                            lambda states: types.positions(
                                x=states.array[:, 1, :], y=states.array[:, 2, :]
                            )
                        ),
                        weight=1.0,
                    ),
                    inputs := data.control_input_batch(
                        np.zeros((T := 1, D_u := 1, M := 1))
                    ),
                    states := data.state_batch(
                        array(
                            [[[phi := 5.0], [x := 5.0 + offset], [y := 5.0 - offset]]],
                            shape=(T, D_x := 3, M),
                        )
                    ),
                    expected_error := 0.0,
                )
                for offset in (-1.0, 0.0, 1.0)
            ],
            *[  # Ego progress ahead/behind reference
                (
                    lag_cost := costs.tracking.lag(
                        reference=trajectory.waypoints(
                            points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)),
                            path_length=10.0,
                        ),
                        path_parameter_extractor=path_parameter_extractor(
                            lambda states: types.path_parameters(states.array[:, 0, :])
                        ),
                        position_extractor=(
                            lambda states: types.positions(
                                x=states.array[:, 1, :], y=states.array[:, 2, :]
                            )
                        ),
                        weight=1.0,
                    ),
                    inputs := data.control_input_batch(
                        np.zeros((T := 1, D_u := 1, M := 1))
                    ),
                    states := data.state_batch(
                        array([[[phi], [x_position], [0.0]]], shape=(T, D_x := 3, M))
                    ),
                    # Horizontal path at y=0, heading=0, reference at x=phi
                    # Lag error = -cos(0)*(x - x_ref) - sin(0)*(y - y_ref) = phi - x
                    expected_error := phi - x_position,
                )
                for x_position in (7.0, 3.0)
                for phi in (5.0,)
            ],
        ]

    @mark.parametrize(
        ["lag_cost", "inputs", "states", "expected_error"],
        [
            *cases(
                costs=costs.numpy,
                trajectory=trajectory.numpy,
                path_parameter_extractor=numpy_path_parameter_extractor,
                types=types.numpy,
                data=data.numpy,
            ),
            *cases(
                costs=costs.jax,
                trajectory=trajectory.jax,
                path_parameter_extractor=jax_path_parameter_extractor,
                types=types.jax,
                data=data.jax,
            ),
        ],
    )
    def test[InputBatchT, StateBatchT](
        self,
        lag_cost: LagCost[InputBatchT, StateBatchT, Error],
        inputs: InputBatchT,
        states: StateBatchT,
        expected_error: float,
    ) -> None:
        error = lag_cost.error(inputs=inputs, states=states)
        assert np.allclose(error, expected_error, atol=1e-10)


class test_that_lag_error_handles_multiple_timesteps_and_rollouts:
    @staticmethod
    def cases(
        costs, trajectory, path_parameter_extractor, types, data
    ) -> Sequence[tuple]:
        return [
            (
                lag_cost := costs.tracking.lag(
                    reference=trajectory.waypoints(
                        points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)),
                        path_length=10.0,
                    ),
                    path_parameter_extractor=path_parameter_extractor(
                        lambda states: types.path_parameters(states.array[:, 0, :])
                    ),
                    position_extractor=(
                        lambda states: types.positions(
                            x=states.array[:, 1, :], y=states.array[:, 2, :]
                        )
                    ),
                    weight=1.0,
                ),
                inputs := data.control_input_batch(
                    np.zeros((T := 2, D_u := 1, M := 3))
                ),
                states := data.state_batch(
                    array(
                        [
                            # phi (path param), x, y for 3 rollouts
                            [
                                [phi_00 := 2.0, phi_01 := 5.0, phi_02 := 8.0],
                                [x_00 := 3.0, x_01 := 6.0, x_02 := 7.0],
                                [0.0, 0.0, 0.0],
                            ],
                            [
                                [phi_10 := 3.0, phi_11 := 6.0, phi_12 := 9.0],
                                [x_10 := 2.0, x_11 := 7.0, x_12 := 10.0],
                                [0.0, 0.0, 0.0],
                            ],
                        ],
                        shape=(T, D_x := 3, M),
                    )
                ),
                expected_errors := array(
                    [
                        [phi_00 - x_00, phi_01 - x_01, phi_02 - x_02],
                        [phi_10 - x_10, phi_11 - x_11, phi_12 - x_12],
                    ],
                    shape=(T, M),
                ),
            ),
        ]

    @mark.parametrize(
        ["lag_cost", "inputs", "states", "expected_errors"],
        [
            *cases(
                costs=costs.numpy,
                trajectory=trajectory.numpy,
                path_parameter_extractor=numpy_path_parameter_extractor,
                types=types.numpy,
                data=data.numpy,
            ),
            *cases(
                costs=costs.jax,
                trajectory=trajectory.jax,
                path_parameter_extractor=jax_path_parameter_extractor,
                types=types.jax,
                data=data.jax,
            ),
        ],
    )
    def test[InputBatchT, StateBatchT](
        self,
        lag_cost: LagCost[InputBatchT, StateBatchT, Error],
        inputs: InputBatchT,
        states: StateBatchT,
        expected_errors: Array,
    ) -> None:
        error = lag_cost.error(inputs=inputs, states=states)
        assert np.allclose(error, expected_errors, atol=1e-10)
