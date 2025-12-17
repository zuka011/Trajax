from trajax import (
    costs,
    trajectory,
    types,
    ContouringCost,
    ControlInputBatch,
    StateBatch,
    Costs,
    Error,
)

from numtypes import array, Array

import numpy as np
import jax.numpy as jnp

from tests.dsl import mppi as data, clear_type
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


@mark.parametrize(
    ["contouring_cost", "inputs", "states", "expected_error"],
    [
        (
            contouring_cost := costs.numpy.tracking.contouring(
                reference=trajectory.numpy.waypoints(
                    points=array([[0.0, 0.0], [10.0, 10.0]], shape=(2, 2)),
                    path_length=(path_length := 10.0),
                ),
                path_parameter_extractor=numpy_path_parameter_extractor(
                    lambda states: types.numpy.path_parameters(
                        np.asarray(states)[:, 0, :]
                    )
                ),
                position_extractor=(
                    lambda states: types.numpy.positions(
                        x=np.asarray(states)[:, 1, :],
                        y=np.asarray(states)[:, 2, :],
                    )
                ),
                weight=1.0,
            ),
            inputs := data.numpy.control_input_batch(
                np.zeros((T := 1, D_u := 1, M := 1))
            ),
            states := data.numpy.state_batch(
                array([[[phi := 5.0], [x := 7.5], [y := 7.5]]], shape=(T, D_x := 3, M))
            ),
            expected_error := 0.0,
        ),
        *[
            (
                contouring_cost := costs.numpy.tracking.contouring(
                    reference=trajectory.numpy.waypoints(
                        points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)),
                        path_length=10.0,
                    ),
                    path_parameter_extractor=numpy_path_parameter_extractor(
                        lambda states: types.numpy.path_parameters(
                            np.asarray(states)[:, 0, :]
                        )
                    ),
                    position_extractor=(
                        lambda states: types.numpy.positions(
                            x=np.asarray(states)[:, 1, :],
                            y=np.asarray(states)[:, 2, :],
                        )
                    ),
                    weight=1.0,
                ),
                inputs := data.numpy.control_input_batch(
                    np.zeros((T := 1, D_u := 1, M := 1))
                ),
                states := data.numpy.state_batch(
                    array([[[phi], [5.0], [lateral_deviation]]], shape=(T, D_x := 3, M))
                ),
                expected_error := -lateral_deviation,  # Left of path is negative error
            )
            for lateral_deviation in (2.0, -0.5)
            for phi in (0.0, 5.0)
        ],
        (
            contouring_cost := costs.jax.tracking.contouring(
                reference=trajectory.jax.waypoints(
                    points=array([[00.0, 0.0], [10.0, 10.0]], shape=(2, 2)),
                    path_length=10.0,
                ),
                path_parameter_extractor=jax_path_parameter_extractor(
                    lambda states: types.jax.path_parameters(
                        states.array[:, 0, :],
                        horizon=states.horizon,
                        rollout_count=states.rollout_count,
                    )
                ),
                position_extractor=(
                    lambda states: types.jax.positions(
                        x=states.array[:, 1, :],
                        y=states.array[:, 2, :],
                        horizon=states.horizon,
                        rollout_count=states.rollout_count,
                    )
                ),
                weight=1.0,
            ),
            inputs := data.jax.control_input_batch(
                jnp.zeros((T := 1, D_u := 1, M := 1))
            ),
            states := data.jax.state_batch(
                array([[[phi := 9.0], [x := 5.0], [y := 5.0]]], shape=(T, D_x := 3, M))
            ),
            expected_error := 0.0,
        ),
        *[
            (
                contouring_cost := costs.jax.tracking.contouring(
                    reference=trajectory.jax.waypoints(
                        points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)),
                        path_length=10.0,
                    ),
                    path_parameter_extractor=(
                        lambda states: types.jax.path_parameters(
                            states.array[:, 0, :],
                            horizon=states.horizon,
                            rollout_count=states.rollout_count,
                        )
                    ),
                    position_extractor=(
                        lambda states: types.jax.positions(
                            x=states.array[:, 1, :],
                            y=states.array[:, 2, :],
                            horizon=states.horizon,
                            rollout_count=states.rollout_count,
                        )
                    ),
                    weight=1.0,
                ),
                inputs := data.jax.control_input_batch(
                    jnp.zeros((T := 1, D_u := 1, M := 1))
                ),
                states := data.jax.state_batch(
                    array([[[phi], [5.0], [lateral_deviation]]], shape=(T, D_x := 3, M))
                ),
                expected_error := -lateral_deviation,
            )
            for lateral_deviation in (3.0, -1.5)
            for phi in (0.0, 7.0)
        ],
    ],
)
def test_that_contouring_cost_computes_correct_lateral_error[
    InputsT: ControlInputBatch,
    StatesT: StateBatch,
](
    contouring_cost: ContouringCost[InputsT, StatesT, Costs, Error],
    inputs: InputsT,
    states: StatesT,
    expected_error: float,
) -> None:
    error = contouring_cost.error(inputs=inputs, states=states)
    assert np.allclose(error, expected_error, atol=1e-10)


T = clear_type
M = clear_type


@mark.parametrize(
    ["contouring_cost", "inputs", "states", "expected_errors"],
    [
        (
            contouring_cost := costs.numpy.tracking.contouring(
                reference=trajectory.numpy.waypoints(
                    points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)),
                    path_length=10.0,
                ),
                path_parameter_extractor=numpy_path_parameter_extractor(
                    lambda states: types.numpy.path_parameters(
                        np.asarray(states)[:, 0, :]
                    )
                ),
                position_extractor=(
                    lambda states: types.numpy.positions(
                        x=np.asarray(states)[:, 1, :],
                        y=np.asarray(states)[:, 2, :],
                    )
                ),
                weight=1.0,
            ),
            inputs := data.numpy.control_input_batch(
                np.zeros((T := 2, D_u := 1, M := 3))
            ),
            states := data.numpy.state_batch(
                array(
                    [
                        [[2.0, 5.0, 8.0], [2.0, 5.0, 8.0], [0.5, 1.0, -1.5]],
                        [[3.0, 6.0, 9.0], [3.0, 6.0, 9.0], [1.0, -2.0, 0.0]],
                    ],
                    shape=(T, D_x := 3, M),
                )
            ),
            expected_errors := array([[0.5, 1.0, 1.5], [1.0, 2.0, 0.0]], shape=(T, M)),
        ),
        (
            contouring_cost := costs.jax.tracking.contouring(
                reference=trajectory.jax.waypoints(
                    points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)),
                    path_length=10.0,
                ),
                path_parameter_extractor=jax_path_parameter_extractor(
                    lambda states: types.jax.path_parameters(
                        states.array[:, 0, :],
                        horizon=states.horizon,
                        rollout_count=states.rollout_count,
                    )
                ),
                position_extractor=(
                    lambda states: types.jax.positions(
                        x=states.array[:, 1, :],
                        y=states.array[:, 2, :],
                        horizon=states.horizon,
                        rollout_count=states.rollout_count,
                    )
                ),
                weight=1.0,
            ),
            inputs := data.jax.control_input_batch(
                jnp.zeros((T := 2, D_u := 1, M := 3))
            ),
            states := data.jax.state_batch(
                array(
                    [
                        [[2.0, 5.0, 8.0], [2.0, 5.0, 8.0], [0.5, 1.0, -1.5]],
                        [[3.0, 6.0, 9.0], [3.0, 6.0, 9.0], [1.0, -2.0, 0.0]],
                    ],
                    shape=(T, D_x := 3, M),
                )
            ),
            expected_errors := array([[0.5, 1.0, 1.5], [1.0, 2.0, 0.0]], shape=(T, M)),
        ),
    ],
)
def test_that_contouring_error_handles_multiple_timesteps_and_rollouts[
    InputsT: ControlInputBatch,
    StatesT: StateBatch,
](
    contouring_cost: ContouringCost[InputsT, StatesT, Costs, Error],
    inputs: InputsT,
    states: StatesT,
    expected_errors: Array,
) -> None:
    error = contouring_cost.error(inputs=inputs, states=states)
    assert np.allclose(np.abs(error), expected_errors, atol=1e-10)
