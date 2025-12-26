from typing import Callable
from functools import partial

from trajax import (
    ControlInputBatch,
    StateBatch,
    Costs,
    CostFunction,
    Circles,
    types,
    trajectory,
    costs,
    obstacles,
    distance as distance_measure,
    risk,
)

from numtypes import array

import numpy as np
import jax.numpy as jnp

from tests.dsl import mppi as data, clear_type, stubs
from pytest import mark, approx


@mark.parametrize(
    [
        "cost",
        "transformed_cost",
        "inputs",
        "states",
        "transformed_states",
    ],
    [
        *[  # Translation Invariance Tests
            (
                cost := tracking_cost(
                    reference=trajectory.numpy.line(
                        start=(x_ref_0 := 2.0, y_ref_0 := 2.0),
                        end=(x_ref_f := 6.0, y_ref_f := 9.0),
                        path_length=(length := 2),
                    ),
                    # States are: [x, y, phi, psi]
                    # The path parameter is the third state dimension (phi)
                    # Psi is some made-up variable.
                    path_parameter_extractor=(
                        path_extractor := lambda states: types.numpy.path_parameters(
                            np.asarray(states)[:, 2]
                        )
                    ),
                    position_extractor=(
                        position_extractor := lambda states: types.numpy.positions(
                            x=np.asarray(states)[:, 0],
                            y=np.asarray(states)[:, 1],
                        )
                    ),
                    weight=(k := 2.5),
                ),
                transformed_cost := tracking_cost(
                    reference=trajectory.numpy.line(
                        start=(x_ref_0 + (d_x := 5.0), y_ref_0 + (d_y := -3.0)),
                        end=(x_ref_f + d_x, y_ref_f + d_y),
                        path_length=length,
                    ),
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=k,
                ),
                # Doesn't matter for this cost function.
                inputs := data.numpy.control_input_batch(
                    np.random.uniform(size=(T := 3, 2, M := 2))  # type: ignore
                ),
                states := data.numpy.state_batch(
                    array(
                        np.array(
                            [
                                rollout_1 := [
                                    [2.0, 2.0, 0.0, 15.0],
                                    [4.0, 5.5, 1.0, 16.0],
                                    [6.0, 9.0, 2.0, 17.0],
                                ],
                                rollout_2 := [
                                    [3.0, 3.0, 0.0, 15.0],
                                    [5.0, 6.5, 0.5, 16.0],
                                    [7.0, 10.0, 1.5, 17.0],
                                ],
                            ]
                        )
                        .transpose(1, 2, 0)
                        .tolist(),
                        shape=(T, D_x := 4, M),
                    )
                ),
                transformed_states := data.numpy.state_batch(
                    array(
                        np.array(
                            [
                                shifted_rollout_1 := [
                                    [x + d_x, y + d_y, phi, psi]
                                    for x, y, phi, psi in rollout_1
                                ],
                                shifted_rollout_2 := [
                                    [x + d_x, y + d_y, phi, psi]
                                    for x, y, phi, psi in rollout_2
                                ],
                            ]
                        )
                        .transpose(1, 2, 0)
                        .tolist(),
                        shape=(T, D_x, M),
                    )
                ),
            )
            for tracking_cost in (
                costs.numpy.tracking.contouring,
                costs.numpy.tracking.lag,
            )
        ],
        *[  # Rotation Invariance Tests
            (
                cost := costs.numpy.tracking.contouring(
                    reference=trajectory.numpy.line(
                        start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                    ),
                    path_parameter_extractor=(
                        path_extractor := lambda states: types.numpy.path_parameters(
                            np.asarray(states)[:, 2]
                        )
                    ),
                    position_extractor=(
                        position_extractor := lambda states: types.numpy.positions(
                            x=np.asarray(states)[:, 0], y=np.asarray(states)[:, 1]
                        )
                    ),
                    weight=(k := 1.0),
                ),
                transformed_cost := costs.numpy.tracking.contouring(
                    reference=trajectory.numpy.line(
                        start=(0.0, 0.0), end=(0.0, 10.0), path_length=1
                    ),
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=k,
                ),
                inputs := data.numpy.control_input_batch(np.zeros((1, 2, 1))),
                states := data.numpy.state_batch(
                    # Robot at (5, 1), 1m lateral offset from X-axis line
                    np.array([[[5.0, 1.0, 0.0, 0.0]]]).transpose(1, 2, 0)
                ),
                transformed_states := data.numpy.state_batch(
                    # Robot at (-1, 5), 1m lateral offset from Y-axis line (rotated +90 deg)
                    np.array([[[-1.0, 5.0, 0.0, 2]]]).transpose(1, 2, 0)
                ),
            ),
            (
                cost := costs.numpy.tracking.lag(
                    reference=trajectory.numpy.line(
                        start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                    ),
                    path_parameter_extractor=(
                        path_extractor := lambda states: types.numpy.path_parameters(
                            np.asarray(states)[:, 2]
                        )
                    ),
                    position_extractor=(
                        position_extractor := lambda states: types.numpy.positions(
                            x=np.asarray(states)[:, 0], y=np.asarray(states)[:, 1]
                        )
                    ),
                    weight=(k := 1.0),
                ),
                transformed_cost := costs.numpy.tracking.lag(
                    reference=trajectory.numpy.line(
                        start=(0.0, 0.0), end=(0.0, 10.0), path_length=1
                    ),
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=k,
                ),
                inputs := data.numpy.control_input_batch(np.zeros((1, 2, 1))),
                states := data.numpy.state_batch(
                    # Robot at (5, 0) but internal phi=4.0 (1m lag)
                    np.array([[[5.0, 0.0, 4.0, 0.0]]]).transpose(1, 2, 0)
                ),
                transformed_states := data.numpy.state_batch(
                    # Robot at (0, 5) but internal phi=4.0 (1m lag preserved)
                    np.array([[[0.0, 5.0, 4.0, 1]]]).transpose(1, 2, 0)
                ),
            ),
        ],
        *[  # Analogous Tests for JAX Implementation
            (
                cost := tracking_cost(
                    reference=trajectory.jax.line(
                        start=(x_ref_0 := 2.0, y_ref_0 := 2.0),
                        end=(x_ref_f := 6.0, y_ref_f := 9.0),
                        path_length=(length := 2),
                    ),
                    path_parameter_extractor=(
                        path_extractor := lambda states: types.jax.path_parameters(
                            states.array[:, 2],
                            horizon=states.horizon,
                            rollout_count=states.rollout_count,
                        )
                    ),
                    position_extractor=(
                        position_extractor := lambda states: types.jax.positions(
                            x=states.array[:, 0],
                            y=states.array[:, 1],
                            horizon=states.horizon,
                            rollout_count=states.rollout_count,
                        )
                    ),
                    weight=(k := 2.5),
                ),
                transformed_cost := tracking_cost(
                    reference=trajectory.jax.line(
                        start=(x_ref_0 + (d_x := 5.0), y_ref_0 + (d_y := -3.0)),
                        end=(x_ref_f + d_x, y_ref_f + d_y),
                        path_length=length,
                    ),
                    path_parameter_extractor=path_extractor,  # type: ignore
                    position_extractor=position_extractor,  # type: ignore
                    weight=k,
                ),
                inputs := data.jax.control_input_batch(
                    np.random.uniform(size=(T := 3, 2, M := 2))  # type: ignore
                ),
                states := data.jax.state_batch(
                    array(
                        np.array(
                            [
                                rollout_1 := [
                                    [2.0, 2.0, 0.0, 15.0],
                                    [4.0, 5.5, 1.0, 16.0],
                                    [6.0, 9.0, 2.0, 17.0],
                                ],
                                rollout_2 := [
                                    [3.0, 3.0, 0.0, 15.0],
                                    [5.0, 6.5, 0.5, 16.0],
                                    [7.0, 10.0, 1.5, 17.0],
                                ],
                            ]
                        )
                        .transpose(1, 2, 0)
                        .tolist(),
                        shape=(T, D_x := 4, M),
                    )
                ),
                transformed_states := data.jax.state_batch(
                    array(
                        np.array(
                            [
                                shifted_rollout_1 := [
                                    [x + d_x, y + d_y, phi, psi]
                                    for x, y, phi, psi in rollout_1
                                ],
                                shifted_rollout_2 := [
                                    [x + d_x, y + d_y, phi, psi]
                                    for x, y, phi, psi in rollout_2
                                ],
                            ]
                        )
                        .transpose(1, 2, 0)
                        .tolist(),
                        shape=(T, D_x, M),
                    )
                ),
            )
            for tracking_cost in (costs.jax.tracking.contouring, costs.jax.tracking.lag)
        ],
        *[
            (
                cost := costs.jax.tracking.contouring(
                    reference=trajectory.jax.line(
                        start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                    ),
                    path_parameter_extractor=(
                        path_extractor := lambda states: types.jax.path_parameters(
                            states.array[:, 2],
                            horizon=states.horizon,
                            rollout_count=states.rollout_count,
                        )
                    ),
                    position_extractor=(
                        position_extractor := lambda states: types.jax.positions(
                            x=states.array[:, 0],
                            y=states.array[:, 1],
                            horizon=states.horizon,
                            rollout_count=states.rollout_count,
                        )
                    ),
                    weight=(k := 1.0),
                ),
                transformed_cost := costs.jax.tracking.contouring(
                    reference=trajectory.jax.line(
                        start=(0.0, 0.0), end=(0.0, 10.0), path_length=1
                    ),
                    path_parameter_extractor=path_extractor,  # type: ignore
                    position_extractor=position_extractor,  # type: ignore
                    weight=k,
                ),
                inputs := data.jax.control_input_batch(np.zeros((1, 2, 1))),
                states := data.jax.state_batch(
                    array(
                        # Robot at (5, 1), 1m lateral offset from X-axis line
                        np.array([[[5.0, 1.0, 0.0, 0.0]]]).transpose(1, 2, 0).tolist(),
                        shape=(T_1 := 1, D_x := 4, M_1 := 1),
                    )
                ),
                transformed_states := data.jax.state_batch(
                    array(
                        # Robot at (-1, 5), 1m lateral offset from Y-axis line (rotated +90 deg)
                        np.array([[[-1.0, 5.0, 0.0, 2]]]).transpose(1, 2, 0).tolist(),
                        shape=(T_1, D_x, M_1),
                    )
                ),
            ),
            (
                cost := costs.jax.tracking.lag(
                    reference=trajectory.jax.line(
                        start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                    ),
                    path_parameter_extractor=(
                        path_extractor := lambda states: types.jax.path_parameters(
                            states.array[:, 2],
                            horizon=states.horizon,
                            rollout_count=states.rollout_count,
                        )
                    ),
                    position_extractor=(
                        position_extractor := lambda states: types.jax.positions(
                            x=states.array[:, 0],
                            y=states.array[:, 1],
                            horizon=states.horizon,
                            rollout_count=states.rollout_count,
                        )
                    ),
                    weight=(k := 1.0),
                ),
                transformed_cost := costs.jax.tracking.lag(
                    reference=trajectory.jax.line(
                        start=(0.0, 0.0), end=(0.0, 10.0), path_length=1
                    ),
                    path_parameter_extractor=path_extractor,  # type: ignore
                    position_extractor=position_extractor,  # type: ignore
                    weight=k,
                ),
                inputs := data.jax.control_input_batch(np.zeros((1, 2, 1))),
                states := data.jax.state_batch(
                    array(
                        # Robot at (5, 0) but internal phi=4.0 (1m lag)
                        np.array([[[5.0, 0.0, 4.0, 0.0]]]).transpose(1, 2, 0).tolist(),
                        shape=(T_1 := 1, D_x := 4, M_1 := 1),
                    )
                ),
                transformed_states := data.jax.state_batch(
                    array(
                        # Robot at (0, 5) but internal phi=4.0 (1m lag preserved)
                        np.array([[[0.0, 5.0, 4.0, 1.0]]]).transpose(1, 2, 0).tolist(),
                        shape=(T_1, D_x, M_1),
                    )
                ),
            ),
        ],
    ],
)
def test_that_tracking_cost_does_not_depend_on_coordinate_system[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    inputs: ControlInputBatchT,
    states: StateBatchT,
    transformed_states: StateBatchT,
    cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    transformed_cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
) -> None:
    assert np.allclose(
        cost(inputs=inputs, states=states),
        transformed_cost(inputs=inputs, states=transformed_states),
    ), "Tracking cost should be invariant to absolute position shifts."


T = clear_type
M = clear_type


@mark.parametrize(
    [
        "cost",
        "inputs",
        "states",
        "zero_deviation_indices",
        "equal_deviation_indices",
        "increasing_deviation_indices",
    ],
    [
        (
            cost := costs.numpy.tracking.contouring(
                reference=trajectory.numpy.line(
                    start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                ),
                path_parameter_extractor=lambda states: types.numpy.path_parameters(
                    np.asarray(states)[:, 2]
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0],
                    y=np.asarray(states)[:, 1],
                ),
                weight=1.0,
            ),
            inputs := data.numpy.control_input_batch(
                np.random.uniform(size=(T := 2, 2, M := 5))  # type: ignore
            ),
            states := data.numpy.state_batch(
                array(
                    np.array(
                        [
                            # Exactly on point.
                            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.5]],
                            # Slightly off the path
                            [[0.0, 0.5, 0.0], [3.0, -0.5, 0.3]],
                            # Slightly lagging/leading
                            [[-0.5, 0.0, 0.1], [5.5, 0.0, 0.4]],
                            # Slightly off and lagging/leading
                            [[-0.5, -0.5, 0.1], [4.5, 0.5, 0.4]],
                            # Very off the path
                            [[0.0, 2.0, 0.0], [5.0, -2.0, 0.5]],
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T, 3, M),
                )
            ),
            zero_deviation_indices := [(0, 0), (1, 0), (0, 2), (1, 2)],
            equal_deviation_indices := [(0, 1), (1, 1), (0, 3), (1, 3)],
            increasing_deviation_indices := [(0, 0), (0, 1), (0, 4)],
        ),
        (
            cost := costs.jax.tracking.contouring(
                reference=trajectory.jax.line(
                    start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                ),
                path_parameter_extractor=lambda states: types.jax.path_parameters(
                    states.array[:, 2],
                    horizon=states.horizon,
                    rollout_count=states.rollout_count,
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0],
                    y=states.array[:, 1],
                    horizon=states.horizon,
                    rollout_count=states.rollout_count,
                ),
                weight=1.0,
            ),
            inputs := data.jax.control_input_batch(
                np.random.uniform(size=(T := 2, 2, M := 5))  # type: ignore
            ),
            states := data.jax.state_batch(
                array(
                    np.array(
                        [
                            # Exactly on point.
                            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.5]],
                            # Slightly off the path
                            [[0.0, 0.5, 0.0], [3.0, -0.5, 0.3]],
                            # Slightly lagging/leading
                            [[-0.5, 0.0, 0.1], [5.5, 0.0, 0.4]],
                            # Slightly off and lagging/leading
                            [[-0.5, -0.5, 0.1], [4.5, 0.5, 0.4]],
                            # Very off the path
                            [[0.0, 2.0, 0.0], [5.0, -2.0, 0.5]],
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T, 3, M),
                )
            ),
            zero_deviation_indices := [(0, 0), (1, 0), (0, 2), (1, 2)],
            equal_deviation_indices := [(0, 1), (1, 1), (0, 3), (1, 3)],
            increasing_deviation_indices := [(0, 0), (0, 1), (0, 4)],
        ),
    ],
)
def test_that_contouring_cost_increases_with_lateral_deviation[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    inputs: ControlInputBatchT,
    states: StateBatchT,
    cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    zero_deviation_indices: list[tuple[int, int]],
    equal_deviation_indices: list[tuple[int, int]],
    increasing_deviation_indices: list[tuple[int, int]],
) -> None:
    J = np.asarray(cost(inputs=inputs, states=states))

    assert all([J[i, j] == 0.0 for i, j in zero_deviation_indices]), (
        f"Cost should be zero when lateral deviation is zero. Got costs: {J}"
    )
    assert all(
        [
            J[i, j] == J[next_i, next_j]
            for (i, j), (next_i, next_j) in zip(
                equal_deviation_indices, equal_deviation_indices[1:]
            )
        ]
    ), f"Cost should be equal for same lateral deviation. Got costs: {J}"
    assert all(
        [
            J[i, j] < J[next_i, next_j]
            for (i, j), (next_i, next_j) in zip(
                increasing_deviation_indices, increasing_deviation_indices[1:]
            )
        ]
    ), f"Cost should increase with increasing lateral deviation. Got costs: {J}"


@mark.parametrize(
    [
        "cost",
        "inputs",
        "states",
        "zero_deviation_indices",
        "equal_deviation_indices",
        "increasing_deviation_indices",
    ],
    [
        (
            cost := costs.numpy.tracking.lag(
                reference=trajectory.numpy.line(
                    start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                ),
                path_parameter_extractor=lambda states: types.numpy.path_parameters(
                    np.asarray(states)[:, 2]
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0],
                    y=np.asarray(states)[:, 1],
                ),
                weight=1.0,
            ),
            inputs := data.numpy.control_input_batch(
                np.random.uniform(size=(T := 2, 2, M := 5))  # type: ignore
            ),
            states := data.numpy.state_batch(
                array(
                    np.array(
                        [
                            # Exactly on point.
                            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.5]],
                            # Slightly off the path (perpendicular, no lag)
                            [[0.0, 0.5, 0.0], [3.0, -0.5, 0.3]],
                            # Slightly lagging (0.5 behind)
                            [[-0.5, 0.0, 0.0], [4.5, 0.0, 0.5]],
                            # Slightly lagging with perpendicular offset (still 0.5 lag)
                            [[-0.5, -0.5, 0.0], [4.5, 0.5, 0.5]],
                            # Very lagging (2.0 behind)
                            [[-2.0, 0.0, 0.0], [3.0, 0.0, 0.5]],
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T, 3, M),
                )
            ),
            zero_deviation_indices := [(0, 0), (1, 0), (0, 1), (1, 1)],
            equal_deviation_indices := [(0, 2), (1, 2), (0, 3), (1, 3)],
            increasing_deviation_indices := [(0, 0), (0, 2), (0, 4)],
        ),
        (
            cost := costs.jax.tracking.lag(
                reference=trajectory.jax.line(
                    start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                ),
                path_parameter_extractor=lambda states: types.jax.path_parameters(
                    states.array[:, 2],
                    horizon=states.horizon,
                    rollout_count=states.rollout_count,
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0],
                    y=states.array[:, 1],
                    horizon=states.horizon,
                    rollout_count=states.rollout_count,
                ),
                weight=1.0,
            ),
            inputs := data.jax.control_input_batch(
                np.random.uniform(size=(T := 2, 2, M := 5))  # type: ignore
            ),
            states := data.jax.state_batch(
                array(
                    np.array(
                        [
                            # Exactly on point.
                            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.5]],
                            # Slightly off the path (perpendicular, no lag)
                            [[0.0, 0.5, 0.0], [3.0, -0.5, 0.3]],
                            # Slightly lagging (0.5 behind)
                            [[-0.5, 0.0, 0.0], [4.5, 0.0, 0.5]],
                            # Slightly lagging with perpendicular offset (still 0.5 lag)
                            [[-0.5, -0.5, 0.0], [4.5, 0.5, 0.5]],
                            # Very lagging (2.0 behind)
                            [[-2.0, 0.0, 0.0], [3.0, 0.0, 0.5]],
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T, 3, M),
                )
            ),
            zero_deviation_indices := [(0, 0), (1, 0), (0, 1), (1, 1)],
            equal_deviation_indices := [(0, 2), (1, 2), (0, 3), (1, 3)],
            increasing_deviation_indices := [(0, 0), (0, 2), (0, 4)],
        ),
    ],
)
def test_that_lag_cost_increases_with_longitudinal_deviation[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    inputs: ControlInputBatchT,
    states: StateBatchT,
    cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    zero_deviation_indices: list[tuple[int, int]],
    equal_deviation_indices: list[tuple[int, int]],
    increasing_deviation_indices: list[tuple[int, int]],
) -> None:
    J = np.asarray(cost(inputs=inputs, states=states))

    assert all([J[i, j] == 0.0 for i, j in zero_deviation_indices]), (
        f"Cost should be zero when longitudinal deviation is zero. Got costs: {J}"
    )
    assert all(
        [
            J[i, j] == J[next_i, next_j]
            for (i, j), (next_i, next_j) in zip(
                equal_deviation_indices, equal_deviation_indices[1:]
            )
        ]
    ), f"Cost should be equal for same longitudinal deviation. Got costs: {J}"
    assert all(
        [
            J[i, j] < J[next_i, next_j]
            for (i, j), (next_i, next_j) in zip(
                increasing_deviation_indices, increasing_deviation_indices[1:]
            )
        ]
    ), f"Cost should increase with increasing longitudinal deviation. Got costs: {J}"


T = clear_type
M = clear_type


@mark.parametrize(
    ["inputs", "states", "create_cost"],
    [
        *[
            (
                inputs := data.numpy.control_input_batch(
                    np.random.uniform(size=(T := 2, 2, M := 2))  # type: ignore
                ),
                states := data.numpy.state_batch(
                    np.random.uniform(size=(T, 3, M))  # type: ignore
                ),
                create_cost := lambda weight: tracking_cost(
                    reference=trajectory.numpy.line(
                        start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                    ),
                    path_parameter_extractor=(
                        lambda states: types.numpy.path_parameters(
                            np.asarray(states)[:, 2]
                        )
                    ),
                    position_extractor=(
                        lambda states: types.numpy.positions(
                            x=np.asarray(states)[:, 0],
                            y=np.asarray(states)[:, 1],
                        )
                    ),
                    weight=weight,
                ),
            )
            for tracking_cost in (
                costs.numpy.tracking.contouring,
                costs.numpy.tracking.lag,
            )
        ],
        *[
            (
                inputs := data.jax.control_input_batch(
                    np.random.uniform(size=(T := 2, 2, M := 2))  # type: ignore
                ),
                states := data.jax.state_batch(
                    np.random.uniform(size=(T, 3, M))  # type: ignore
                ),
                create_cost := lambda weight: tracking_cost(
                    reference=trajectory.jax.line(
                        start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                    ),
                    path_parameter_extractor=lambda states: types.jax.path_parameters(
                        states.array[:, 2],
                        horizon=states.horizon,
                        rollout_count=states.rollout_count,
                    ),
                    position_extractor=(
                        lambda states: types.jax.positions(
                            x=states.array[:, 0],
                            y=states.array[:, 1],
                            horizon=states.horizon,
                            rollout_count=states.rollout_count,
                        )
                    ),
                    weight=weight,
                ),
            )
            for tracking_cost in (costs.jax.tracking.contouring, costs.jax.tracking.lag)
        ],
        *[
            (
                inputs := data.numpy.control_input_batch(
                    np.ones((T := 2, D_u := 2, M := 2))
                ),
                states := data.numpy.state_batch(
                    np.random.uniform(size=(T, 3, M))  # type: ignore
                ),
                create_cost := partial(
                    lambda weight, *, weight_time_step: costs.numpy.tracking.progress(
                        path_velocity_extractor=(lambda u: np.asarray(u)[:, 1]),
                        # Increasing either weight or time step size should increase cost
                        time_step_size=weight if weight_time_step else 1.0,
                        weight=weight if not weight_time_step else 1.0,
                    ),
                    weight_time_step=weight_time_step,
                ),
            )
            for weight_time_step in (True, False)
        ],
        *[
            (
                inputs := data.jax.control_input_batch(
                    np.ones((T := 2, D_u := 2, M := 2))
                ),
                states := data.jax.state_batch(
                    np.random.uniform(size=(T, 3, M))  # type: ignore
                ),
                create_cost := partial(
                    lambda weight, *, weight_time_step: costs.jax.tracking.progress(
                        path_velocity_extractor=lambda u: u.array[:, 1],
                        time_step_size=weight if weight_time_step else 1.0,
                        weight=weight if not weight_time_step else 1.0,
                    ),
                    weight_time_step=weight_time_step,
                ),
            )
            for weight_time_step in (True, False)
        ],
        (
            inputs := data.numpy.control_input_batch(
                np.random.uniform(size=(T := 3, D_u := 2, M := 2))  # type: ignore
            ),
            states := data.numpy.state_batch(
                np.random.uniform(size=(T, D_x := 4, M))  # type: ignore
            ),
            create_cost := partial(
                lambda weight, *, expected_states: costs.numpy.safety.collision(
                    obstacle_states=stubs.ObstacleStateProvider.returns(
                        obstacle_states := data.numpy.obstacle_states(
                            x=np.random.uniform(size=(T := 3, K := 3)),  # type: ignore
                            y=np.random.uniform(size=(T, K)),  # type: ignore
                        )
                    ),
                    sampler=stubs.ObstacleStateSampler.returns(
                        obstacle_state_samples := data.numpy.obstacle_state_samples(
                            x=np.random.uniform(size=(T, K, N := 1)),  # type: ignore
                            y=np.random.uniform(size=(T, K, N)),  # type: ignore
                        ),
                        when_obstacle_states_are=obstacle_states,
                        and_sample_count_is=N,
                    ),
                    distance=stubs.DistanceExtractor.returns(
                        data.numpy.distance(
                            np.random.uniform(size=(T, V := 3, M := 2, N))  # type: ignore
                        ),
                        when_states_are=expected_states,
                        and_obstacle_states_are=obstacle_state_samples,
                    ),
                    distance_threshold=array([1.0, 2.0, 3.0], shape=(V,)),
                    weight=weight,
                ),
                expected_states=states,
            ),
        ),
        (
            inputs := data.jax.control_input_batch(
                np.random.uniform(size=(T := 3, D_u := 2, M := 2))  # type: ignore
            ),
            jax_states := data.jax.state_batch(
                np.random.uniform(size=(T, D_x := 4, M))  # type: ignore
            ),
            create_cost := partial(
                lambda weight, *, expected_states: costs.jax.safety.collision(
                    obstacle_states=stubs.ObstacleStateProvider.returns(
                        obstacle_states := data.jax.obstacle_states(
                            x=np.random.uniform(size=(T := 3, K := 3)),  # type: ignore
                            y=np.random.uniform(size=(T, K)),  # type: ignore
                        )
                    ),
                    sampler=stubs.ObstacleStateSampler.returns(
                        obstacle_state_samples := data.jax.obstacle_state_samples(
                            x=np.random.uniform(size=(T, K, N := 1)),  # type: ignore
                            y=np.random.uniform(size=(T, K, N)),  # type: ignore
                        ),
                        when_obstacle_states_are=obstacle_states,
                        and_sample_count_is=N,
                    ),
                    distance=stubs.DistanceExtractor.returns(
                        data.jax.distance(
                            np.random.uniform(size=(T, V := 3, M := 2, N))  # type: ignore
                        ),
                        when_states_are=expected_states,
                        and_obstacle_states_are=obstacle_state_samples,
                    ),
                    distance_threshold=array([1.0, 2.0, 3.0], shape=(V,)),
                    weight=weight,
                ),
                expected_states=jax_states,
            ),
        ),
    ],
)
def test_that_cost_increases_with_weight[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    inputs: ControlInputBatchT,
    states: StateBatchT,
    create_cost: Callable[
        [float], CostFunction[ControlInputBatchT, StateBatchT, CostsT]
    ],
) -> None:
    J_low = np.abs(create_cost(1.0)(inputs=inputs, states=states))
    J_high = np.abs(create_cost(20.0)(inputs=inputs, states=states))

    assert np.all(J_high > J_low), (
        f"Absolute value of cost should increase with weight. Got low weight costs: {J_low}, "
        f"high weight costs: {J_high}"
    )


@mark.parametrize(
    ["cost", "inputs_slow", "inputs_fast", "states"],
    [
        (
            cost := costs.numpy.tracking.progress(
                path_velocity_extractor=lambda u: np.asarray(u)[:, 1],
                time_step_size=0.1,
                weight=1.0,
            ),
            inputs_slow := data.numpy.control_input_batch(
                np.full((T := 2, D_u := 2, M := 2), 2.0)  # type: ignore
            ),
            inputs_fast := data.numpy.control_input_batch(
                np.full((T, D_u, M), 4.0)  # type: ignore
            ),
            # Irrelevant for this test.
            states := data.numpy.state_batch(np.zeros((T, D_x := 2, M))),
        ),
        (
            cost := costs.jax.tracking.progress(
                path_velocity_extractor=lambda u: u.array[:, 1],
                time_step_size=0.1,
                weight=1.0,
            ),
            inputs_slow := data.jax.control_input_batch(
                np.full((T := 2, D_u := 2, M := 2), 2.0)  # type: ignore
            ),
            inputs_fast := data.jax.control_input_batch(
                np.full((T, D_u, M), 4.0)  # type: ignore
            ),
            states := data.jax.state_batch(np.zeros((T, D_x := 2, M))),
        ),
    ],
)
def test_that_progress_cost_rewards_velocity[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    inputs_slow: ControlInputBatchT,
    inputs_fast: ControlInputBatchT,
    states: StateBatchT,
) -> None:
    J_slow = np.asarray(cost(inputs=inputs_slow, states=states))
    J_fast = np.asarray(cost(inputs=inputs_fast, states=states))

    assert np.all(0 > J_slow), "Progress cost should be negative (a reward)."
    assert np.all(J_slow > J_fast), "Higher path velocity should yield higher reward."


@mark.parametrize(
    [
        "cost",
        "inputs",
        "states",
        "zero_jerk_indices",
        "equal_jerk_indices",
        "increasing_jerk_indices",
    ],
    [
        (
            cost := costs.numpy.comfort.control_smoothing(
                weights=array([1.0, 0.5, 0.2], shape=(D_u := 3,))
            ),
            inputs := data.numpy.control_input_batch(
                array(
                    np.array(
                        [
                            # Constant High: [10, 10, 10] -> Diff is [0, 0]
                            [[10, 10, 10], [10, 10, 10], [10, 10, 10]],
                            # Oscillating: [1, -1, 1] -> Diff is [-2, 2] -> Non-zero Cost
                            [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                            # Constant Zero: [0, 0, 0] -> Diff is [0, 0]
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            # Wildly Changing: [5, -5, 5] -> Diff is [-10, 10] -> Higher Cost
                            [[5.0, 5.0, 5.0], [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]],
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T := 3, D_u, M := 4),
                )
            ),
            states := data.numpy.state_batch(
                np.zeros((T, D_x := 2, M))  # type: ignore
            ),
            zero_jerk_indices := [(1, 0), (2, 0), (1, 2), (2, 2)],
            equal_jerk_indices := [(1, 1), (2, 1)],
            increasing_jerk_indices := [(1, 0), (1, 1), (1, 3)],
        ),
        (
            cost := costs.jax.comfort.control_smoothing(
                weights=array([1.0, 0.5, 0.2], shape=(D_u := 3,))
            ),
            inputs := data.jax.control_input_batch(
                array(
                    np.array(
                        [
                            # Constant High: [10, 10, 10] -> Diff is [0, 0]
                            [[10, 10, 10], [10, 10, 10], [10, 10, 10]],
                            # Oscillating: [1, -1, 1] -> Diff is [-2, 2] -> Non-zero Cost
                            [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                            # Constant Zero: [0, 0, 0] -> Diff is [0, 0]
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                            # Wildly Changing: [5, -5, 5] -> Diff is [-10, 10] -> Higher Cost
                            [[5.0, 5.0, 5.0], [-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]],
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T := 3, D_u, M := 4),
                )
            ),
            states := data.jax.state_batch(
                np.zeros((T, D_x := 2, M))  # type: ignore
            ),
            zero_jerk_indices := [(1, 0), (2, 0), (1, 2), (2, 2)],
            equal_jerk_indices := [(1, 1), (2, 1)],
            increasing_jerk_indices := [(1, 0), (1, 1), (1, 3)],
        ),
    ],
)
def test_that_control_smoothing_cost_increases_with_higher_jerk[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    inputs: ControlInputBatchT,
    states: StateBatchT,
    cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    zero_jerk_indices: list[tuple[int, int]],
    equal_jerk_indices: list[tuple[int, int]],
    increasing_jerk_indices: list[tuple[int, int]],
) -> None:
    J = np.asarray(cost(inputs=inputs, states=states))

    assert np.allclose(J[0, :], 0.0), (
        f"Cost for the first time step should be zero as there is no prior input. Got costs: {J}"
    )
    assert all([J[i, j] == 0.0 for i, j in zero_jerk_indices]), (
        f"Cost should be zero for constant inputs. Got costs: {J}"
    )
    assert all(
        [
            J[i, j] == J[next_i, next_j]
            for (i, j), (next_i, next_j) in zip(
                equal_jerk_indices, equal_jerk_indices[1:]
            )
        ]
    ), f"Cost should be equal for same input changes. Got costs: {J}"
    assert all(
        [
            J[i, j] < J[next_i, next_j]
            for (i, j), (next_i, next_j) in zip(
                increasing_jerk_indices, increasing_jerk_indices[1:]
            )
        ]
    ), f"Cost should increase with higher input changes. Got costs: {J}"


@mark.parametrize(
    [
        "cost",
        "inputs",
        "states",
        "low_cost_indices",
        "high_cost_indices",
    ],
    [
        (
            cost := costs.numpy.comfort.control_smoothing(
                weights=array([1.0, 10.0], shape=(D_u := 2,))
            ),
            inputs := data.numpy.control_input_batch(
                array(
                    np.array(
                        [
                            # Batch 0: Change only in Dim 0 (Low Weight)
                            # [0,0] -> [1,0] -> [2,0] ...
                            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                            # Batch 1: Change only in Dim 1 (High Weight)
                            # [0,0] -> [0,1] -> [0,2] ...
                            # Note: The magnitude of change (1.0) is identical to Batch 0.
                            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T := 3, D_u, M := 2),
                )
            ),
            states := data.numpy.state_batch(
                np.zeros((T, D_x := 2, M))  # type: ignore
            ),
            low_cost_indices := [(1, 0), (2, 0)],
            high_cost_indices := [(1, 1), (2, 1)],
        ),
        (
            cost := costs.jax.comfort.control_smoothing(
                weights=jnp.array([1.0, 10.0]), dimensions=(D_u := 2)
            ),
            inputs := data.jax.control_input_batch(
                array(
                    np.array(
                        [
                            # Batch 0: Change only in Dim 0 (Low Weight)
                            # [0,0] -> [1,0] -> [2,0] ...
                            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                            # Batch 1: Change only in Dim 1 (High Weight)
                            # [0,0] -> [0,1] -> [0,2] ...
                            # Note: The magnitude of change (1.0) is identical to Batch 0.
                            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
                        ]
                    )
                    .transpose(1, 2, 0)
                    .tolist(),
                    shape=(T := 3, D_u, M := 2),
                )
            ),
            states := data.jax.state_batch(
                np.zeros((T, D_x := 2, M))  # type: ignore
            ),
            low_cost_indices := [(1, 0), (2, 0)],
            high_cost_indices := [(1, 1), (2, 1)],
        ),
    ],
)
def test_that_control_smoothing_cost_respects_dimension_weights[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    inputs: ControlInputBatchT,
    states: StateBatchT,
    cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    low_cost_indices: list[tuple[int, int]],
    high_cost_indices: list[tuple[int, int]],
) -> None:
    J = np.asarray(cost(inputs=inputs, states=states))

    assert all(
        [
            J[low_i, low_j] < J[high_i, high_j]
            for (low_i, low_j), (high_i, high_j) in zip(
                low_cost_indices, high_cost_indices
            )
        ]
    ), (
        f"Cost should be higher for changes in dimensions with higher weights. "
        f"Got costs: {J}"
    )


@mark.parametrize(
    [
        "cost",
        "inputs",
        "states",
        "cost_order",
        "non_zero_cost_indices",
        "zero_cost_indices",
    ],
    [
        (
            cost := costs.numpy.safety.collision(
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    obstacle_states := data.numpy.obstacle_states(
                        x=np.random.uniform(size=(T := 4, K := 1)),  # type: ignore
                        y=np.random.uniform(size=(T, K)),  # type: ignore
                    )
                ),
                sampler=stubs.ObstacleStateSampler.returns(
                    obstacle_state_samples := data.numpy.obstacle_state_samples(
                        x=np.random.uniform(size=(T, K, N := 1)),  # type: ignore
                        y=np.random.uniform(size=(T, K, N)),  # type: ignore
                    ),
                    when_obstacle_states_are=obstacle_states,
                    and_sample_count_is=N,
                ),
                distance=stubs.DistanceExtractor.returns(
                    data.numpy.distance(
                        array(  # Robot has V = 2 parts.
                            [
                                # Collisions
                                [
                                    [[-0.8], [-0.7], [-0.6], [-0.5]],
                                    [[-0.6], [-0.5], [-0.4], [-0.3]],
                                ],
                                # Near Collision (close to threshold)
                                [
                                    [[-0.2], [-0.1], [0.0], [0.1]],
                                    [[0.5], [0.6], [0.7], [0.8]],
                                ],
                                # Middle Distance (just below threshold)
                                [
                                    [[0.6], [0.7], [0.8], [0.9]],
                                    [[0.8], [0.85], [0.9], [0.95]],
                                ],
                                # Very Far (well above threshold)
                                [
                                    [[2.0], [2.5], [3.0], [5.0]],
                                    [[3.0], [3.5], [4.0], [10.0]],
                                ],
                            ],
                            shape=(T, V := 2, M := 4, N),
                        )
                    ),
                    when_states_are=(
                        states := data.numpy.state_batch(
                            np.random.uniform(size=(T, D_x := 2, M)),  # type: ignore
                        )
                    ),
                    and_obstacle_states_are=obstacle_state_samples,
                ),
                distance_threshold=array([1.0, 1.0], shape=(V,)),
                weight=2.0,
            ),
            inputs := data.numpy.control_input_batch(np.zeros((T, D_u := 3, M))),
            states,
            cost_order := [
                *[(0, 0), (0, 1), (0, 2), (0, 3)],  # Collisions
                *[(1, 0), (1, 1), (1, 2), (1, 3)],  # Near Collision
                *[(2, 0), (2, 1), (2, 2), (2, 3)],  # Middle Distance
                *[(3, 0), (3, 1), (3, 2), (3, 3)],  # Very Far
            ],
            non_zero_cost_indices := [(0, 0), (1, 2), (2, 3)],
            zero_cost_indices := [(3, 0), (3, 1), (3, 2), (3, 3)],
        ),
        (
            cost := costs.jax.safety.collision(
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    jax_obstacle_states := data.jax.obstacle_states(
                        x=np.random.uniform(size=(T := 4, K := 1)),  # type: ignore
                        y=np.random.uniform(size=(T, K)),  # type: ignore
                    )
                ),
                sampler=stubs.ObstacleStateSampler.returns(
                    jax_obstacle_state_samples := data.jax.obstacle_state_samples(
                        x=np.random.uniform(size=(T, K, N := 1)),  # type: ignore
                        y=np.random.uniform(size=(T, K, N)),  # type: ignore
                    ),
                    when_obstacle_states_are=jax_obstacle_states,
                    and_sample_count_is=N,
                ),
                distance=stubs.DistanceExtractor.returns(
                    data.jax.distance(
                        array(  # Robot has V = 2 parts.
                            [
                                # Collisions
                                [
                                    [[-0.8], [-0.7], [-0.6], [-0.5]],
                                    [[-0.6], [-0.5], [-0.4], [-0.3]],
                                ],
                                # Near Collision (close to threshold)
                                [
                                    [[-0.2], [-0.1], [0.0], [0.1]],
                                    [[0.5], [0.6], [0.7], [0.8]],
                                ],
                                # Middle Distance (just below threshold)
                                [
                                    [[0.6], [0.7], [0.8], [0.9]],
                                    [[0.8], [0.85], [0.9], [0.95]],
                                ],
                                # Very Far (well above threshold)
                                [
                                    [[2.0], [2.5], [3.0], [5.0]],
                                    [[3.0], [3.5], [4.0], [10.0]],
                                ],
                            ],
                            shape=(T, V := 2, M := 4, N),
                        )
                    ),
                    when_states_are=(
                        states := data.jax.state_batch(
                            np.random.uniform(size=(T, D_x := 2, M)),  # type: ignore
                        )
                    ),
                    and_obstacle_states_are=jax_obstacle_state_samples,
                ),
                distance_threshold=array([1.0, 1.0], shape=(V,)),
                weight=2.0,
            ),
            inputs := data.jax.control_input_batch(np.zeros((T, D_u := 3, M))),
            states,
            cost_order := [
                *[(0, 0), (0, 1), (0, 2), (0, 3)],  # Collisions
                *[(1, 0), (1, 1), (1, 2), (1, 3)],  # Near Collision
                *[(2, 0), (2, 1), (2, 2), (2, 3)],  # Middle Distance
                *[(3, 0), (3, 1), (3, 2), (3, 3)],  # Very Far
            ],
            non_zero_cost_indices := [(0, 0), (1, 2), (2, 3)],
            zero_cost_indices := [(3, 0), (3, 1), (3, 2), (3, 3)],
        ),
    ],
)
def test_that_collision_cost_decreases_with_distance[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    inputs: ControlInputBatchT,
    states: StateBatchT,
    cost_order: list[tuple[int, int]],
    non_zero_cost_indices: list[tuple[int, int]],
    zero_cost_indices: list[tuple[int, int]],
) -> None:
    J = np.asarray(cost(inputs=inputs, states=states))

    assert all(
        [
            J[t, m] >= J[next_t, next_m]
            for (t, m), (next_t, next_m) in zip(cost_order, cost_order[1:])
        ]
    ), (
        f"Collision cost should decrease with increasing distance to obstacles. Got costs: {J}"
    )
    assert all([J[t, m] > 0.0 for (t, m) in non_zero_cost_indices]), (
        f"Cost should be non-zero when distance is less than threshold. Got costs: {J}"
    )
    assert all([J[t, m] == approx(0.0, abs=1e-3) for (t, m) in zero_cost_indices]), (
        f"Cost should be zero when distance is significantly above threshold. Got costs: {J}"
    )


@mark.parametrize(
    ["cost", "inputs", "states", "cost_order"],
    [
        (
            cost := costs.numpy.safety.collision(
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    obstacle_states := data.numpy.obstacle_states(
                        x=np.random.uniform(size=(T := 4, K := 1)),  # type: ignore
                        y=np.random.uniform(size=(T, K)),  # type: ignore
                    )
                ),
                sampler=stubs.ObstacleStateSampler.returns(
                    obstacle_state_samples := data.numpy.obstacle_state_samples(
                        x=np.random.uniform(size=(T, K, N := 1)),  # type: ignore
                        y=np.random.uniform(size=(T, K, N)),  # type: ignore
                    ),
                    when_obstacle_states_are=obstacle_states,
                    and_sample_count_is=N,
                ),
                distance=stubs.DistanceExtractor.returns(
                    data.numpy.distance(
                        array(
                            [
                                # Only v=2 contributes
                                [[[20], [10]], [[100.0], [20.0]], [[0.6], [0.65]]],
                                # v=1 and v=2 contribute
                                [[[0.3], [0.35]], [[0.3], [0.35]], [[0.3], [0.35]]],
                                # All parts contribute
                                [[[0.1], [0.15]], [[0.1], [0.15]], [[0.1], [0.15]]],
                                # All parts contribute
                                [
                                    [[-0.1], [-0.15]],
                                    [[-0.1], [-0.15]],
                                    [[-0.1], [-0.15]],
                                ],
                            ],
                            shape=(T, V := 3, M := 2, N),
                        )
                    ),
                    when_states_are=(
                        states := data.numpy.state_batch(
                            np.random.uniform(size=(T, D_x := 1, M)),  # type: ignore
                        )
                    ),
                    and_obstacle_states_are=obstacle_state_samples,
                ),
                distance_threshold=array([0.25, 0.5, 0.75], shape=(V,)),
                weight=2.0,
            ),
            inputs := data.numpy.control_input_batch(np.zeros((T, D_u := 3, M))),
            states,
            cost_order := [(2, 0), (2, 1), (1, 0), (1, 1), (0, 0), (0, 1)],
        ),
        (
            cost := costs.jax.safety.collision(
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    jax_obstacle_states := data.jax.obstacle_states(
                        x=np.random.uniform(size=(T := 4, K := 1)),  # type: ignore
                        y=np.random.uniform(size=(T, K)),  # type: ignore
                    )
                ),
                sampler=stubs.ObstacleStateSampler.returns(
                    jax_obstacle_state_samples := data.jax.obstacle_state_samples(
                        x=np.random.uniform(size=(T, K, N := 1)),  # type: ignore
                        y=np.random.uniform(size=(T, K, N)),  # type: ignore
                    ),
                    when_obstacle_states_are=jax_obstacle_states,
                    and_sample_count_is=N,
                ),
                distance=stubs.DistanceExtractor.returns(
                    data.jax.distance(
                        array(
                            [
                                # Only v=2 contributes
                                [[[20], [10]], [[100.0], [20.0]], [[0.6], [0.65]]],
                                # v=1 and v=2 contribute
                                [[[0.3], [0.35]], [[0.3], [0.35]], [[0.3], [0.35]]],
                                # All parts contribute
                                [[[0.1], [0.15]], [[0.1], [0.15]], [[0.1], [0.15]]],
                                # All parts contribute
                                [
                                    [[-0.1], [-0.15]],
                                    [[-0.1], [-0.15]],
                                    [[-0.1], [-0.15]],
                                ],
                            ],
                            shape=(T, V := 3, M := 2, N),
                        )
                    ),
                    when_states_are=(
                        states := data.jax.state_batch(
                            np.random.uniform(size=(T, D_x := 1, M)),  # type: ignore
                        )
                    ),
                    and_obstacle_states_are=jax_obstacle_state_samples,
                ),
                distance_threshold=array([0.25, 0.5, 0.75], shape=(V,)),
                weight=2.0,
            ),
            inputs := data.jax.control_input_batch(np.zeros((T, D_u := 3, M))),
            states,
            cost_order := [(2, 0), (2, 1), (1, 0), (1, 1), (0, 0), (0, 1)],
        ),
    ],
)
def test_that_collision_cost_uses_different_thresholds_for_different_parts[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    inputs: ControlInputBatchT,
    states: StateBatchT,
    cost_order: list[tuple[int, int]],
) -> None:
    J = np.asarray(cost(inputs=inputs, states=states))

    assert all(
        [
            J[t, m] > J[next_t, next_m]
            for (t, m), (next_t, next_m) in zip(cost_order, cost_order[1:])
        ]
    ), f"Collision cost should reflect different thresholds per part. Got costs: {J}"


@mark.parametrize(
    ["cost", "inputs", "states"],
    [
        (
            cost := costs.numpy.safety.collision(
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    obstacle_states := data.numpy.obstacle_states(
                        x=np.random.uniform(size=(T := 2, K := 0)),  # type: ignore
                        y=np.random.uniform(size=(T, K)),  # type: ignore
                    )
                ),
                sampler=stubs.ObstacleStateSampler.returns(
                    obstacle_state_samples := data.numpy.obstacle_state_samples(
                        x=np.random.uniform(size=(T, K, N := 1)),  # type: ignore
                        y=np.random.uniform(size=(T, K, N)),  # type: ignore
                    ),
                    when_obstacle_states_are=obstacle_states,
                    and_sample_count_is=N,
                ),
                distance=stubs.DistanceExtractor.returns(
                    data.numpy.distance(
                        np.full((T, V := 2, M := 3, N), np.inf)  # type: ignore
                    ),
                    when_states_are=(
                        states := data.numpy.state_batch(
                            np.random.uniform(size=(T, D_x := 1, M)),  # type: ignore
                        )
                    ),
                    and_obstacle_states_are=obstacle_state_samples,
                ),
                distance_threshold=array([0.5, 0.5], shape=(V,)),
                weight=10.0,
            ),
            inputs := data.numpy.control_input_batch(np.zeros((T, D_u := 2, M))),
            states,
        ),
        (
            cost := costs.jax.safety.collision(
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    jax_obstacle_states := data.jax.obstacle_states(
                        x=np.random.uniform(size=(T := 2, K := 0)),  # type: ignore
                        y=np.random.uniform(size=(T, K)),  # type: ignore
                    )
                ),
                sampler=stubs.ObstacleStateSampler.returns(
                    jax_obstacle_state_samples := data.jax.obstacle_state_samples(
                        x=np.random.uniform(size=(T, K, N := 1)),  # type: ignore
                        y=np.random.uniform(size=(T, K, N)),  # type: ignore
                    ),
                    when_obstacle_states_are=jax_obstacle_states,
                    and_sample_count_is=N,
                ),
                distance=stubs.DistanceExtractor.returns(
                    data.jax.distance(
                        np.full((T, V := 2, M := 3, N), np.inf)  # type: ignore
                    ),
                    when_states_are=(
                        states := data.jax.state_batch(
                            np.random.uniform(size=(T, D_x := 1, M)),  # type: ignore
                        )
                    ),
                    and_obstacle_states_are=jax_obstacle_state_samples,
                ),
                distance_threshold=array([0.5, 0.5], shape=(V,)),
                weight=10.0,
            ),
            inputs := data.jax.control_input_batch(np.zeros((T, D_u := 2, M))),
            states,
        ),
    ],
)
def test_that_collision_cost_is_zero_when_no_obstacles_are_present[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    inputs: ControlInputBatchT,
    states: StateBatchT,
) -> None:
    J = np.asarray(cost(inputs=inputs, states=states))

    assert np.allclose(J, 0.0), (
        f"Collision cost should be zero when no obstacles are present. Got costs: {J}"
    )


@mark.parametrize(
    ["inputs", "states", "create_cost"],
    [
        (
            inputs := data.numpy.control_input_batch(
                np.zeros((T := 10, D_u := 2, M := 5))
            ),
            states := data.numpy.state_batch(
                array(
                    np.tile(  # type: ignore
                        [
                            [[-1.0], [-2.0], [np.pi / 4]],
                        ],
                        (T, 1, M),
                    ),
                    shape=(T, D_x := 3, M),
                )
            ),
            create_cost := lambda variance: costs.numpy.safety.collision(
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    obstacle_states := data.numpy.obstacle_states(
                        x=(rng := np.random.default_rng(seed=42)).uniform(
                            low=-1.0, high=1.0, size=(T := 10, K := 5)
                        ),
                        y=rng.uniform(low=0.0, high=2.0, size=(T, K)),
                        heading=rng.uniform(low=-np.pi, high=np.pi, size=(T, K)),
                        covariance=array(
                            # Each covariance matrix is diagonal with `variance` on the diagonal.
                            (variance * np.eye(D_O := types.obstacle.D_O))[
                                None, ..., None
                            ]
                            * np.ones((T, D_O, D_O, K)),
                            shape=(T, D_O, D_O, K),
                        ),
                    )
                ),
                sampler=obstacles.sampler.numpy.gaussian(),
                distance=distance_measure.numpy.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([2.5], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([2.5], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.numpy.positions(
                        x=np.asarray(states)[:, 0],
                        y=np.asarray(states)[:, 1],
                    ),
                    heading_extractor=lambda states: types.numpy.headings(
                        theta=np.asarray(states)[:, 2]
                    ),
                ),
                distance_threshold=array([0.5], shape=(V,)),
                weight=10.0,
                metric=risk.numpy.mean_variance(gamma=2.0, sample_count=100),
            ),
        ),
        (
            inputs := data.jax.control_input_batch(
                np.zeros((T := 10, D_u := 2, M := 5))
            ),
            states := data.jax.state_batch(
                array(
                    np.tile(  # type: ignore
                        [
                            [[-1.0], [-2.0], [np.pi / 4]],
                        ],
                        (T, 1, M),
                    ),
                    shape=(T, D_x := 3, M),
                )
            ),
            create_cost := lambda variance: costs.jax.safety.collision(
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    data.jax.obstacle_states(
                        x=(rng := np.random.default_rng(seed=42)).uniform(
                            low=-1.0, high=1.0, size=(T := 10, K := 5)
                        ),
                        y=rng.uniform(low=0.0, high=2.0, size=(T, K)),
                        heading=rng.uniform(low=-np.pi, high=np.pi, size=(T, K)),
                        covariance=array(
                            (variance * np.eye(D_O := types.obstacle.D_O))[
                                None, ..., None
                            ]
                            * np.ones((T, D_O, D_O, K)),
                            shape=(T, D_O, D_O, K),
                        ),
                    )
                ),
                sampler=obstacles.sampler.jax.gaussian(),
                distance=distance_measure.jax.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([2.5], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([2.5], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.jax.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                        horizon=states.horizon,
                        rollout_count=states.rollout_count,
                    ),
                    heading_extractor=lambda states: types.jax.headings(
                        theta=states.array[:, 2],
                        horizon=states.horizon,
                        rollout_count=states.rollout_count,
                    ),
                ),
                distance_threshold=array([0.5], shape=(V,)),
                weight=10.0,
                metric=risk.jax.mean_variance(gamma=2.0, sample_count=100),
            ),
        ),
    ],
)
def test_that_collision_cost_increases_with_higher_obstacle_state_uncertainty[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    inputs: ControlInputBatchT,
    states: StateBatchT,
    create_cost: Callable[
        [float], CostFunction[ControlInputBatchT, StateBatchT, CostsT]
    ],
) -> None:
    J_low = np.asarray(create_cost(0.01)(inputs=inputs, states=states))
    J_high = np.asarray(create_cost(0.5)(inputs=inputs, states=states))

    assert np.all(J_high > J_low), (
        f"Collision cost should increase with higher position uncertainty. "
        f"Got costs: {J_low} (low variance) vs {J_high} (high variance)"
    )
