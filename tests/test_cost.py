from trajax import (
    ControlInputBatch,
    StateBatch,
    Costs,
    CostFunction,
    types,
    trajectory,
    costs,
)

import numpy as np
from numtypes import array

from tests.dsl import mppi as data, clear_type
from pytest import mark


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
                    np.random.uniform(size=(T := 3, 2, M := 2))
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
                    np.random.uniform(size=(T := 3, 2, M := 2))
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
                np.random.uniform(size=(T := 2, 2, M := 5))
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
                np.random.uniform(size=(T := 2, 2, M := 5))
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
                np.random.uniform(size=(T := 2, 2, M := 5))
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
                np.random.uniform(size=(T := 2, 2, M := 5))
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
    [
        "inputs",
        "states",
        "low_weight_cost",
        "high_weight_cost",
    ],
    [
        *[
            (
                inputs := data.numpy.control_input_batch(
                    np.random.uniform(size=(T := 2, 2, M := 2))
                ),
                states := data.numpy.state_batch(
                    np.random.uniform(size=(T, 3, M))  # type: ignore
                ),
                low_weight_cost := tracking_cost(
                    reference=(
                        reference := trajectory.numpy.line(
                            start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                        )
                    ),
                    path_parameter_extractor=(
                        path_parameter_extractor
                        := lambda states: types.numpy.path_parameters(
                            np.asarray(states)[:, 2]
                        )
                    ),
                    position_extractor=(
                        position_extractor := lambda states: types.numpy.positions(
                            x=np.asarray(states)[:, 0],
                            y=np.asarray(states)[:, 1],
                        )
                    ),
                    weight=1.0,
                ),
                high_weight_cost := tracking_cost(
                    reference=reference,
                    path_parameter_extractor=path_parameter_extractor,
                    position_extractor=position_extractor,
                    weight=10.0,
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
                    np.random.uniform(size=(T := 2, 2, M := 2))
                ),
                states := data.jax.state_batch(
                    np.random.uniform(size=(T, 3, M))  # type: ignore
                ),
                low_weight_cost := tracking_cost(
                    reference=(
                        reference := trajectory.jax.line(
                            start=(0.0, 0.0), end=(10.0, 0.0), path_length=1
                        )
                    ),
                    path_parameter_extractor=(
                        path_parameter_extractor
                        := lambda states: types.jax.path_parameters(
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
                    weight=1.0,
                ),
                high_weight_cost := tracking_cost(
                    reference=reference,
                    path_parameter_extractor=path_parameter_extractor,  # type: ignore
                    position_extractor=position_extractor,  # type: ignore
                    weight=10.0,
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
                low_weight_cost := costs.numpy.tracking.progress(
                    path_velocity_extractor=(
                        path_velocity_extractor := lambda u: np.asarray(u)[:, 1]
                    ),
                    time_step_size=low_time_step_size,
                    weight=low_weight,
                ),
                high_weight_cost := costs.numpy.tracking.progress(
                    path_velocity_extractor=path_velocity_extractor,
                    time_step_size=high_time_step_size,
                    weight=high_weight,
                ),
            )
            for (low_time_step_size, high_time_step_size, low_weight, high_weight) in (
                (0.1, 0.1, 1.0, 10.0),
                # Increasing the time step size should also increase cost
                (0.1, 1.0, 1.0, 1.0),
            )
        ],
        *[
            (
                inputs := data.jax.control_input_batch(
                    np.ones((T := 2, D_u := 2, M := 2))
                ),
                states := data.jax.state_batch(
                    np.random.uniform(size=(T, 3, M))  # type: ignore
                ),
                low_weight_cost := costs.jax.tracking.progress(
                    path_velocity_extractor=(
                        path_velocity_extractor := lambda u: u.array[:, 1]
                    ),
                    time_step_size=low_time_step_size,
                    weight=low_weight,
                ),
                high_weight_cost := costs.jax.tracking.progress(
                    path_velocity_extractor=path_velocity_extractor,  # type: ignore
                    time_step_size=high_time_step_size,
                    weight=high_weight,
                ),
            )
            for (low_time_step_size, high_time_step_size, low_weight, high_weight) in (
                (0.1, 0.1, 1.0, 10.0),
                # Increasing the time step size should also increase cost
                (0.1, 1.0, 1.0, 1.0),
            )
        ],
    ],
)
def test_that_cost_increases_with_weight[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
](
    inputs: ControlInputBatchT,
    states: StateBatchT,
    low_weight_cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
    high_weight_cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
) -> None:
    J_low = np.abs(low_weight_cost(inputs=inputs, states=states))
    J_high = np.abs(high_weight_cost(inputs=inputs, states=states))

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
                np.full((T := 2, D_u := 2, M := 2), 2.0)
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
                np.full((T := 2, D_u := 2, M := 2), 2.0)
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
