from typing import Callable, Sequence
from functools import partial

from faran import (
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

from tests.dsl import mppi as data, stubs
from pytest import mark, approx


class test_that_tracking_cost_does_not_depend_on_coordinate_system:
    @staticmethod
    def cases(trajectory, types, data, costs) -> Sequence[tuple]:
        return [
            *[  # Translation Invariance Tests
                (
                    cost := tracking_cost(
                        reference=trajectory.line(
                            start=(x_ref_0 := 2.0, y_ref_0 := 2.0),
                            end=(x_ref_f := 6.0, y_ref_f := 9.0),
                            path_length=(length := 2.0),
                        ),
                        # States are: [x, y, phi, psi]
                        # The path parameter is the third state dimension (phi)
                        # Psi is some made-up variable.
                        path_parameter_extractor=(
                            path_extractor := lambda states: types.path_parameters(
                                states.array[:, 2]
                            )
                        ),
                        position_extractor=(
                            position_extractor := lambda states: types.positions(
                                x=states.array[:, 0],
                                y=states.array[:, 1],
                            )
                        ),
                        weight=(k := 2.5),
                    ),
                    transformed_cost := tracking_cost(
                        reference=trajectory.line(
                            start=(x_ref_0 + (d_x := 5.0), y_ref_0 + (d_y := -3.0)),
                            end=(x_ref_f + d_x, y_ref_f + d_y),
                            path_length=length,
                        ),
                        path_parameter_extractor=path_extractor,
                        position_extractor=position_extractor,
                        weight=k,
                    ),
                    # Doesn't matter for this cost function.
                    inputs := data.control_input_batch(
                        np.random.uniform(size=(T := 3, 2, M := 2))
                    ),
                    states := data.state_batch(
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
                    transformed_states := data.state_batch(
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
                    costs.tracking.contouring,
                    costs.tracking.lag,
                )
            ],
            *[  # Rotation Invariance Tests
                (
                    cost := costs.tracking.contouring(
                        reference=trajectory.line(
                            start=(0.0, 0.0), end=(10.0, 0.0), path_length=1.0
                        ),
                        path_parameter_extractor=(
                            path_extractor := lambda states: types.path_parameters(
                                states.array[:, 2]
                            )
                        ),
                        position_extractor=(
                            position_extractor := lambda states: types.positions(
                                x=states.array[:, 0], y=states.array[:, 1]
                            )
                        ),
                        weight=(k := 1.0),
                    ),
                    transformed_cost := costs.tracking.contouring(
                        reference=trajectory.line(
                            start=(0.0, 0.0), end=(0.0, 10.0), path_length=1.0
                        ),
                        path_parameter_extractor=path_extractor,
                        position_extractor=position_extractor,
                        weight=k,
                    ),
                    inputs := data.control_input_batch(np.zeros((1, 2, 1))),
                    states := data.state_batch(
                        # Robot at (5, 1), 1m lateral offset from X-axis line
                        np.array([[[5.0, 1.0, 0.0, 0.0]]]).transpose(1, 2, 0)
                    ),
                    transformed_states := data.state_batch(
                        # Robot at (-1, 5), 1m lateral offset from Y-axis line (rotated +90 deg)
                        np.array([[[-1.0, 5.0, 0.0, 2]]]).transpose(1, 2, 0)
                    ),
                ),
                (
                    cost := costs.tracking.lag(
                        reference=trajectory.line(
                            start=(0.0, 0.0), end=(10.0, 0.0), path_length=1.0
                        ),
                        path_parameter_extractor=(
                            path_extractor := lambda states: types.path_parameters(
                                states.array[:, 2]
                            )
                        ),
                        position_extractor=(
                            position_extractor := lambda states: types.positions(
                                x=states.array[:, 0], y=states.array[:, 1]
                            )
                        ),
                        weight=(k := 1.0),
                    ),
                    transformed_cost := costs.tracking.lag(
                        reference=trajectory.line(
                            start=(0.0, 0.0), end=(0.0, 10.0), path_length=1.0
                        ),
                        path_parameter_extractor=path_extractor,
                        position_extractor=position_extractor,
                        weight=k,
                    ),
                    inputs := data.control_input_batch(np.zeros((1, 2, 1))),
                    states := data.state_batch(
                        # Robot at (5, 0) but internal phi=4.0 (1m lag)
                        np.array([[[5.0, 0.0, 4.0, 0.0]]]).transpose(1, 2, 0)
                    ),
                    transformed_states := data.state_batch(
                        # Robot at (0, 5) but internal phi=4.0 (1m lag preserved)
                        np.array([[[0.0, 5.0, 4.0, 1]]]).transpose(1, 2, 0)
                    ),
                ),
            ],
        ]

    @mark.parametrize(
        [
            "cost",
            "transformed_cost",
            "inputs",
            "states",
            "transformed_states",
        ],
        [
            *cases(
                trajectory=trajectory.numpy,
                types=types.numpy,
                data=data.numpy,
                costs=costs.numpy,
            ),
            *cases(
                trajectory=trajectory.jax,
                types=types.jax,
                data=data.jax,
                costs=costs.jax,
            ),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
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


class test_that_contouring_cost_increases_with_lateral_deviation:
    @staticmethod
    def cases(trajectory, types, data, costs) -> Sequence[tuple]:
        return [
            (
                cost := costs.tracking.contouring(
                    reference=trajectory.line(
                        start=(0.0, 0.0), end=(10.0, 0.0), path_length=1.0
                    ),
                    path_parameter_extractor=lambda states: types.path_parameters(
                        states.array[:, 2]
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    weight=1.0,
                ),
                inputs := data.control_input_batch(
                    np.random.uniform(size=(T := 2, 2, M := 5))
                ),
                states := data.state_batch(
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
        ]

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
            *cases(
                trajectory=trajectory.numpy,
                types=types.numpy,
                data=data.numpy,
                costs=costs.numpy,
            ),
            *cases(
                trajectory=trajectory.jax,
                types=types.jax,
                data=data.jax,
                costs=costs.jax,
            ),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
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


class test_that_lag_cost_increases_with_longitudinal_deviation:
    @staticmethod
    def cases(trajectory, types, data, costs) -> Sequence[tuple]:
        return [
            (
                cost := costs.tracking.lag(
                    reference=trajectory.line(
                        start=(0.0, 0.0), end=(10.0, 0.0), path_length=1.0
                    ),
                    path_parameter_extractor=lambda states: types.path_parameters(
                        states.array[:, 2]
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    weight=1.0,
                ),
                inputs := data.control_input_batch(
                    np.random.uniform(size=(T := 2, 2, M := 5))  # type: ignore
                ),
                states := data.state_batch(
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
        ]

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
            *cases(
                trajectory=trajectory.numpy,
                types=types.numpy,
                data=data.numpy,
                costs=costs.numpy,
            ),
            *cases(
                trajectory=trajectory.jax,
                types=types.jax,
                data=data.jax,
                costs=costs.jax,
            ),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
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
        ), (
            f"Cost should increase with increasing longitudinal deviation. Got costs: {J}"
        )


class test_that_cost_increases_with_weight:
    @staticmethod
    def cases(trajectory, types, data, costs) -> Sequence[tuple]:
        return [
            *[
                (
                    inputs := data.control_input_batch(
                        np.random.uniform(size=(T := 2, 2, M := 2))
                    ),
                    states := data.state_batch(np.random.uniform(size=(T, 3, M))),
                    create_cost := lambda weight: tracking_cost(
                        reference=trajectory.line(
                            start=(0.0, 0.0), end=(10.0, 0.0), path_length=1.0
                        ),
                        path_parameter_extractor=(
                            lambda states: types.path_parameters(states.array[:, 2])
                        ),
                        position_extractor=(
                            lambda states: types.positions(
                                x=states.array[:, 0],
                                y=states.array[:, 1],
                            )
                        ),
                        weight=weight,
                    ),
                )
                for tracking_cost in (costs.tracking.contouring, costs.tracking.lag)
            ],
            *[
                (
                    inputs := data.control_input_batch(
                        np.ones((T := 2, D_u := 2, M := 2))
                    ),
                    states := data.state_batch(np.random.uniform(size=(T, 3, M))),
                    create_cost := partial(
                        lambda weight, *, weight_time_step: costs.tracking.progress(
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
            (
                inputs := data.control_input_batch(
                    np.random.uniform(size=(T := 3, D_u := 2, M := 2))
                ),
                states := data.state_batch(np.random.uniform(size=(T, D_x := 4, M))),
                create_cost := partial(
                    lambda weight, *, expected_states: costs.safety.collision(
                        obstacle_states=stubs.ObstacleStateProvider.returns(
                            obstacle_states := data.obstacle_2d_poses(
                                x=np.random.uniform(size=(T := 3, K := 3)),
                                y=np.random.uniform(size=(T, K)),
                            )
                        ),
                        sampler=stubs.ObstacleStateSampler.returns(
                            obstacle_state_samples := data.obstacle_2d_pose_samples(
                                x=np.random.uniform(size=(T, K, N := 1)),
                                y=np.random.uniform(size=(T, K, N)),
                            ),
                            when_obstacle_states_are=obstacle_states,
                            and_sample_count_is=N,
                        ),
                        distance=stubs.DistanceExtractor.returns(
                            data.distance(
                                np.random.default_rng(0).uniform(
                                    size=(T, V := 3, M := 2, N)
                                )
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
                inputs := data.control_input_batch(
                    np.random.uniform(size=(T := 2, D_u := 2, M := 2))
                ),
                states := data.state_batch(np.random.uniform(size=(T, D_x := 4, M))),
                create_cost := lambda weight, states=states: costs.safety.boundary(
                    distance=stubs.BoundaryDistanceExtractor.returns(
                        data.boundary_distance(
                            np.random.default_rng(0).uniform(size=(T, M))
                        ),
                        when_states_are=states,
                    ),
                    distance_threshold=2.0,
                    weight=weight,
                ),
            ),
            *[
                (
                    inputs := data.control_input_batch(
                        array(
                            [
                                [[1.0, 2.0], [1.5, 2.5]],
                                [[3.0, 4.0], [3.5, 4.5]],
                            ],
                            shape=(T := 2, D_u := 2, M := 2),
                        )
                    ),
                    states := data.state_batch(np.zeros((T, D_x := 2, M))),
                    create_cost := (
                        lambda weight, i=i, T=T, D_u=D_u: costs.comfort.control_effort(
                            weights=array(
                                [  # Increasing either weight should increase cost.
                                    weight if i == 0 else 1.0,
                                    weight if i == 1 else 1.0,
                                ],
                                shape=(D_u,),
                            ),
                        )
                    ),
                )
                for i in range(2)
            ],
        ]

    @mark.parametrize(
        ["inputs", "states", "create_cost"],
        [
            *cases(
                trajectory=trajectory.numpy,
                types=types.numpy,
                data=data.numpy,
                costs=costs.numpy,
            ),
            *cases(
                trajectory=trajectory.jax,
                types=types.jax,
                data=data.jax,
                costs=costs.jax,
            ),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
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


class test_that_progress_cost_rewards_velocity:
    @staticmethod
    def cases(data, costs) -> Sequence[tuple]:
        return [
            (
                cost := costs.tracking.progress(
                    path_velocity_extractor=lambda u: np.asarray(u)[:, 1],
                    time_step_size=0.1,
                    weight=1.0,
                ),
                inputs_slow := data.control_input_batch(
                    np.full((T := 2, D_u := 2, M := 2), 2.0)  # type: ignore
                ),
                inputs_fast := data.control_input_batch(
                    np.full((T, D_u, M), 4.0)  # type: ignore
                ),
                # Irrelevant for this test.
                states := data.state_batch(np.zeros((T, D_x := 2, M))),
            ),
        ]

    @mark.parametrize(
        ["cost", "inputs_slow", "inputs_fast", "states"],
        [
            *cases(data=data.numpy, costs=costs.numpy),
            *cases(data=data.jax, costs=costs.jax),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
        cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
        inputs_slow: ControlInputBatchT,
        inputs_fast: ControlInputBatchT,
        states: StateBatchT,
    ) -> None:
        J_slow = np.asarray(cost(inputs=inputs_slow, states=states))
        J_fast = np.asarray(cost(inputs=inputs_fast, states=states))

        assert np.all(0 > J_slow), "Progress cost should be negative (a reward)."
        assert np.all(J_slow > J_fast), (
            "Higher path velocity should yield higher reward."
        )


class test_that_control_smoothing_cost_increases_with_higher_jerk:
    @staticmethod
    def cases(data, costs) -> Sequence[tuple]:
        return [
            (
                cost := costs.comfort.control_smoothing(
                    weights=array([1.0, 0.5, 0.2], shape=(D_u := 3,))
                ),
                inputs := data.control_input_batch(
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
                states := data.state_batch(np.zeros((T, D_x := 2, M))),
                zero_jerk_indices := [(1, 0), (2, 0), (1, 2), (2, 2)],
                equal_jerk_indices := [(1, 1), (2, 1)],
                increasing_jerk_indices := [(1, 0), (1, 1), (1, 3)],
            ),
        ]

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
            *cases(data=data.numpy, costs=costs.numpy),
            *cases(data=data.jax, costs=costs.jax),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
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


class test_that_control_smoothing_cost_respects_dimension_weights:
    @staticmethod
    def cases(data, costs) -> Sequence[tuple]:
        return [
            (
                cost := costs.comfort.control_smoothing(
                    weights=array([1.0, 10.0], shape=(D_u := 2,))
                ),
                inputs := data.control_input_batch(
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
                states := data.state_batch(np.zeros((T, D_x := 2, M))),
                low_cost_indices := [(1, 0), (2, 0)],
                high_cost_indices := [(1, 1), (2, 1)],
            ),
        ]

    @mark.parametrize(
        [
            "cost",
            "inputs",
            "states",
            "low_cost_indices",
            "high_cost_indices",
        ],
        [
            *cases(data=data.numpy, costs=costs.numpy),
            *cases(data=data.jax, costs=costs.jax),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
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


class test_that_control_effort_cost_increases_with_control_magnitude:
    @staticmethod
    def cases(data, costs) -> Sequence[tuple]:
        return [
            (
                cost := costs.comfort.control_effort(
                    weights=array([1.0, 1.0], shape=(D_u := 2,))
                ),
                inputs := data.control_input_batch(
                    array(
                        np.array(
                            [
                                # M=0: Zero magnitude -> cost should be 0
                                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                                # M=1: Small positive deviation -> small cost
                                [[0.5, 0.5], [1.0, 0.5], [1.0, 1.0]],
                                # M=2: Large positive deviation -> large cost
                                [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]],
                                # M=3: Negative deviation (same magnitude as rollout 1) -> same cost as rollout 1
                                [[-0.5, -0.5], [-1.0, -0.5], [-1.0, -1.0]],
                            ]
                        )
                        .transpose(1, 2, 0)
                        .tolist(),
                        shape=(T := 3, D_u, M := 4),
                    )
                ),
                states := data.state_batch(np.zeros((T, D_x := 2, M))),
                # All time steps of rollout 0
                zero_cost_indices := [(0, 0), (1, 0), (2, 0)],
                increasing_deviation_indices := [(0, 0), (0, 1), (1, 1), (2, 1)],
                equal_deviation_indices := [
                    ((0, 1), (0, 3)),  # T=0: positive vs negative
                    ((1, 1), (1, 3)),  # T=1: positive vs negative
                    ((2, 1), (2, 3)),  # T=2: positive vs negative
                ],
            ),
        ]

    @mark.parametrize(
        [
            "cost",
            "inputs",
            "states",
            "zero_cost_indices",
            "increasing_deviation_indices",
            "equal_deviation_indices",
        ],
        [
            *cases(data=data.numpy, costs=costs.numpy),
            *cases(data=data.jax, costs=costs.jax),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
        cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
        inputs: ControlInputBatchT,
        states: StateBatchT,
        zero_cost_indices: list[tuple[int, int]],
        increasing_deviation_indices: list[tuple[int, int]],
        equal_deviation_indices: list[tuple[tuple[int, int], tuple[int, int]]],
    ) -> None:
        J = np.asarray(cost(inputs=inputs, states=states))

        assert np.all(J >= 0.0), f"Cost values should be non-negative. Got costs: {J}"

        assert all(
            [J[i, j] == approx(0.0, abs=1e-6) for (i, j) in zero_cost_indices]
        ), f"Cost should be zero when input equals nominal. Got costs: {J}"

        assert all(
            [
                J[low_i, low_j] < J[high_i, high_j]
                for (low_i, low_j), (high_i, high_j) in zip(
                    increasing_deviation_indices, increasing_deviation_indices[1:]
                )
            ]
        ), f"Cost should increase with deviation magnitude. Got costs: {J}"

        assert all(
            [
                J[i_1, j_1] == approx(J[i_2, j_2], abs=1e-6)
                for (i_1, j_1), (i_2, j_2) in equal_deviation_indices
            ]
        ), (
            f"Cost should be symmetric for positive and negative deviations. Got costs: {J}"
        )


class test_that_collision_cost_decreases_with_distance:
    @staticmethod
    def cases(data, costs) -> Sequence[tuple]:
        return [
            (
                cost := costs.safety.collision(
                    obstacle_states=stubs.ObstacleStateProvider.returns(
                        obstacle_states := data.obstacle_2d_poses(
                            x=np.random.uniform(size=(T := 4, K := 1)),
                            y=np.random.uniform(size=(T, K)),
                        )
                    ),
                    sampler=stubs.ObstacleStateSampler.returns(
                        obstacle_state_samples := data.obstacle_2d_pose_samples(
                            x=np.random.uniform(size=(T, K, N := 1)),
                            y=np.random.uniform(size=(T, K, N)),
                        ),
                        when_obstacle_states_are=obstacle_states,
                        and_sample_count_is=N,
                    ),
                    distance=stubs.DistanceExtractor.returns(
                        data.distance(
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
                            states := data.state_batch(
                                np.random.uniform(size=(T, D_x := 2, M)),
                            )
                        ),
                        and_obstacle_states_are=obstacle_state_samples,
                    ),
                    distance_threshold=array([1.0, 1.0], shape=(V,)),
                    weight=2.0,
                ),
                inputs := data.control_input_batch(np.zeros((T, D_u := 3, M))),
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
        ]

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
            *cases(data=data.numpy, costs=costs.numpy),
            *cases(data=data.jax, costs=costs.jax),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
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
        assert all(
            [J[t, m] == approx(0.0, abs=1e-3) for (t, m) in zero_cost_indices]
        ), (
            f"Cost should be zero when distance is significantly above threshold. Got costs: {J}"
        )


class test_that_collision_cost_uses_different_thresholds_for_different_parts:
    @staticmethod
    def cases(data, costs) -> Sequence[tuple]:
        return [
            (
                cost := costs.safety.collision(
                    obstacle_states=stubs.ObstacleStateProvider.returns(
                        obstacle_states := data.obstacle_2d_poses(
                            x=np.random.uniform(size=(T := 4, K := 1)),
                            y=np.random.uniform(size=(T, K)),
                        )
                    ),
                    sampler=stubs.ObstacleStateSampler.returns(
                        obstacle_state_samples := data.obstacle_2d_pose_samples(
                            x=np.random.uniform(size=(T, K, N := 1)),
                            y=np.random.uniform(size=(T, K, N)),
                        ),
                        when_obstacle_states_are=obstacle_states,
                        and_sample_count_is=N,
                    ),
                    distance=stubs.DistanceExtractor.returns(
                        data.distance(
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
                            states := data.state_batch(
                                np.random.uniform(size=(T, D_x := 1, M)),
                            )
                        ),
                        and_obstacle_states_are=obstacle_state_samples,
                    ),
                    distance_threshold=array([0.25, 0.5, 0.75], shape=(V,)),
                    weight=2.0,
                ),
                inputs := data.control_input_batch(np.zeros((T, D_u := 3, M))),
                states,
                cost_order := [(2, 0), (2, 1), (1, 0), (1, 1), (0, 0), (0, 1)],
            ),
        ]

    @mark.parametrize(
        ["cost", "inputs", "states", "cost_order"],
        [
            *cases(data=data.numpy, costs=costs.numpy),
            *cases(data=data.jax, costs=costs.jax),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
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
        ), (
            f"Collision cost should reflect different thresholds per part. Got costs: {J}"
        )


class test_that_collision_cost_is_zero_when_no_obstacles_are_present:
    @staticmethod
    def cases(data, costs, risk) -> Sequence[tuple]:
        return [
            (
                cost := costs.safety.collision(
                    obstacle_states=stubs.ObstacleStateProvider.returns(
                        obstacle_states := data.obstacle_2d_poses(
                            x=np.random.uniform(size=(T := 2, K := 0)),
                            y=np.random.uniform(size=(T, K)),
                        )
                    ),
                    sampler=stubs.ObstacleStateSampler.returns(
                        obstacle_state_samples := data.obstacle_2d_pose_samples(
                            x=np.random.uniform(size=(T, K, N := 1)),
                            y=np.random.uniform(size=(T, K, N)),
                        ),
                        when_obstacle_states_are=obstacle_states,
                        and_sample_count_is=N,
                    ),
                    distance=stubs.DistanceExtractor.returns(
                        data.distance(np.full((T, V := 2, M := 3, N), np.inf)),
                        when_states_are=(
                            states := data.state_batch(
                                np.random.uniform(size=(T, D_x := 1, M)),
                            )
                        ),
                        and_obstacle_states_are=obstacle_state_samples,
                    ),
                    distance_threshold=array([0.5, 0.5], shape=(V,)),
                    weight=10.0,
                ),
                inputs := data.control_input_batch(np.zeros((T, D_u := 2, M))),
                states,
            ),
            *[
                (
                    cost := costs.safety.collision(
                        obstacle_states=stubs.ObstacleStateProvider.returns(
                            obstacle_states := data.obstacle_2d_poses(
                                x=np.random.uniform(size=(T := 2, K := 0)),
                                y=np.random.uniform(size=(T, K)),
                            )
                        ),
                        sampler=stubs.ObstacleStateSampler.returns(
                            obstacle_state_samples := data.obstacle_2d_pose_samples(
                                x=np.random.uniform(size=(T, K, N := 1)),
                                y=np.random.uniform(size=(T, K, N)),
                            ),
                            when_obstacle_states_are=obstacle_states,
                            and_sample_count_is=N,
                        ),
                        distance=stubs.DistanceExtractor.returns(
                            data.distance(np.full((T, V := 2, M := 3, N), np.inf)),
                            when_states_are=(
                                states := data.state_batch(
                                    np.random.uniform(size=(T, D_x := 1, M)),
                                )
                            ),
                            and_obstacle_states_are=obstacle_state_samples,
                        ),
                        distance_threshold=array([0.5, 0.5], shape=(V,)),
                        weight=10.0,
                        # Even if a risk metric is specified, cost should be zero as there are no obstacles.
                        metric=metric,
                    ),
                    inputs := data.control_input_batch(np.zeros((T, D_u := 2, M))),
                    states,
                )
                for metric in (
                    risk.expected_value(sample_count=10),
                    risk.mean_variance(gamma=2.0, sample_count=10),
                    risk.var(alpha=0.95, sample_count=10),
                    risk.cvar(alpha=0.95, sample_count=10),
                    risk.entropic_risk(theta=0.5, sample_count=10),
                )
            ],
        ]

    @mark.parametrize(
        ["cost", "inputs", "states"],
        [
            *cases(data=data.numpy, costs=costs.numpy, risk=risk.numpy),
            *cases(data=data.jax, costs=costs.jax, risk=risk.jax),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
        cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
        inputs: ControlInputBatchT,
        states: StateBatchT,
    ) -> None:
        J = np.asarray(cost(inputs=inputs, states=states))

        assert np.allclose(J, 0.0), (
            f"Collision cost should be zero when no obstacles are present. Got costs: {J}"
        )


class test_that_collision_cost_with_uncertainty_is_consistent_with_deterministic_collision_cost:
    @staticmethod
    def cases(
        data,
        costs,
        distance_measure,
        obstacles,
        risk,
        types,
        types_obstacle=types.obstacle,
    ) -> Sequence[tuple]:
        return [
            (
                inputs := data.control_input_batch(
                    np.zeros((T := 10, D_u := 2, M := 5))
                ),
                states := data.state_batch(
                    array(
                        np.tile(
                            [
                                [[-1.0], [-2.0], [np.pi / 4]],
                            ],
                            (T, 1, M),
                        ),
                        shape=(T, D_x := 3, M),
                    )
                ),
                create_cost := lambda metric, variance: costs.safety.collision(
                    obstacle_states=stubs.ObstacleStateProvider.returns(
                        obstacle_states := data.obstacle_2d_poses(
                            x=(rng := np.random.default_rng(seed=42)).uniform(
                                low=-1.0, high=1.0, size=(T := 10, K := 5)
                            ),
                            y=rng.uniform(low=0.0, high=2.0, size=(T, K)),
                            heading=rng.uniform(low=-np.pi, high=np.pi, size=(T, K)),
                            covariance=array(
                                (variance * np.eye(D_O := types_obstacle.POSE_D_O))[
                                    None, ..., None
                                ]
                                * np.ones((T, D_O, D_O, K)),
                                shape=(T, D_O, D_O, K),
                            ),
                        )
                    ),
                    sampler=obstacles.sampler.gaussian(),
                    distance=distance_measure.circles(
                        ego=Circles(
                            origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                            radii=array([2.5], shape=(V,)),
                        ),
                        obstacle=Circles(
                            origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                            radii=array([2.5], shape=(C,)),
                        ),
                        position_extractor=lambda states: types.positions(
                            x=states.array[:, 0],
                            y=states.array[:, 1],
                        ),
                        heading_extractor=lambda states: types.headings(
                            heading=states.array[:, 2]
                        ),
                        obstacle_position_extractor=lambda states: states.positions(),
                        obstacle_heading_extractor=lambda states: states.headings(),
                    ),
                    distance_threshold=array([0.5], shape=(V,)),
                    weight=10.0,
                    metric=metric,
                ),
                no_metric := risk.none(),
                target_metric := metric,
            )
            for metric in (
                risk.expected_value(sample_count=50),
                risk.mean_variance(gamma=0.0, sample_count=50),
                risk.entropic_risk(theta=1e-6, sample_count=50),
            )
        ]

    @mark.parametrize(
        ["inputs", "states", "create_cost", "no_metric", "target_metric"],
        [
            *cases(
                data=data.numpy,
                costs=costs.numpy,
                distance_measure=distance_measure.numpy,
                obstacles=obstacles.numpy,
                risk=risk.numpy,
                types=types.numpy,
            ),
            *cases(
                data=data.jax,
                costs=costs.jax,
                distance_measure=distance_measure.jax,
                obstacles=obstacles.jax,
                risk=risk.jax,
                types=types.jax,
            ),
        ],
    )
    def test_that_costs_are_equal_when_variance_is_close_to_zero[
        ControlInputBatchT,
        StateBatchT,
        CostsT: Costs,
        RiskMetricT,
    ](
        self,
        inputs: ControlInputBatchT,
        states: StateBatchT,
        create_cost: Callable[
            [RiskMetricT, float], CostFunction[ControlInputBatchT, StateBatchT, CostsT]
        ],
        no_metric: RiskMetricT,
        target_metric: RiskMetricT,
    ) -> None:
        no_variance = 1e-9

        # NOTE: we sum over time steps, since some metrics have no meaning per time step.
        J_deterministic = np.asarray(
            create_cost(no_metric, no_variance)(inputs=inputs, states=states)
        ).sum(axis=0)
        J_uncertain = np.asarray(
            create_cost(target_metric, no_variance)(inputs=inputs, states=states)
        ).sum(axis=0)

        assert np.allclose(J_deterministic, J_uncertain, atol=1e-2), (
            f"Collision cost without uncertainty should equal the mean collision cost "
            f"with uncertainty. Got deterministic costs: {J_deterministic} vs uncertain costs: {J_uncertain}"
        )

    @mark.parametrize(
        ["inputs", "states", "create_cost", "no_metric", "target_metric"],
        [
            *cases(
                data=data.numpy,
                costs=costs.numpy,
                distance_measure=distance_measure.numpy,
                obstacles=obstacles.numpy,
                risk=risk.numpy,
                types=types.numpy,
            ),
            *cases(
                data=data.jax,
                costs=costs.jax,
                distance_measure=distance_measure.jax,
                obstacles=obstacles.jax,
                risk=risk.jax,
                types=types.jax,
            ),
        ],
    )
    def test_that_jensen_inequality_is_satisfied_when_variance_is_not_zero[
        ControlInputBatchT,
        StateBatchT,
        CostsT: Costs,
        RiskMetricT,
    ](
        self,
        inputs: ControlInputBatchT,
        states: StateBatchT,
        create_cost: Callable[
            [RiskMetricT, float], CostFunction[ControlInputBatchT, StateBatchT, CostsT]
        ],
        no_metric: RiskMetricT,
        target_metric: RiskMetricT,
    ) -> None:
        non_zero_variance = 0.1

        # NOTE: we sum over time steps, since some metrics have no meaning per time step.
        J_deterministic = np.asarray(
            create_cost(no_metric, non_zero_variance)(inputs=inputs, states=states)
        ).sum(axis=0)
        J_uncertain = np.asarray(
            create_cost(target_metric, non_zero_variance)(inputs=inputs, states=states)
        ).sum(axis=0)

        # We add some tolerance due to sampling noise.
        assert np.all(J_uncertain <= J_deterministic * 1.01), (
            f"E[J(xi)] <= J(E[xi]) should hold for concave cost functions (Jensen's inequality). "
            f"Got deterministic costs: {J_deterministic} vs uncertain costs: {J_uncertain}"
        )


class test_that_collision_cost_increases_with_higher_variance_in_obstacle_state_uncertainty:
    @staticmethod
    def cases(
        data,
        costs,
        distance_measure,
        obstacles,
        risk,
        types,
        types_obstacle=types.obstacle,
    ) -> Sequence[tuple]:
        return [
            (
                inputs := data.control_input_batch(
                    np.zeros((T := 10, D_u := 2, M := 5))
                ),
                states := data.state_batch(
                    array(
                        np.tile(
                            [
                                [[-1.0], [-2.0], [np.pi / 4]],
                            ],
                            (T, 1, M),
                        ),
                        shape=(T, D_x := 3, M),
                    )
                ),
                create_cost := lambda variance: costs.safety.collision(
                    obstacle_states=stubs.ObstacleStateProvider.returns(
                        obstacle_states := data.obstacle_2d_poses(
                            x=(rng := np.random.default_rng(seed=42)).uniform(
                                low=-1.0, high=1.0, size=(T := 10, K := 5)
                            ),
                            y=rng.uniform(low=0.0, high=2.0, size=(T, K)),
                            heading=rng.uniform(low=-np.pi, high=np.pi, size=(T, K)),
                            covariance=array(
                                # Each covariance matrix is diagonal with `variance` on the diagonal.
                                (variance * np.eye(D_O := types_obstacle.POSE_D_O))[
                                    None, ..., None
                                ]
                                * np.ones((T, D_O, D_O, K)),
                                shape=(T, D_O, D_O, K),
                            ),
                        )
                    ),
                    sampler=obstacles.sampler.gaussian(),
                    distance=distance_measure.circles(
                        ego=Circles(
                            origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                            radii=array([2.5], shape=(V,)),
                        ),
                        obstacle=Circles(
                            origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                            radii=array([2.5], shape=(C,)),
                        ),
                        position_extractor=lambda states: types.positions(
                            x=states.array[:, 0],
                            y=states.array[:, 1],
                        ),
                        heading_extractor=lambda states: types.headings(
                            heading=states.array[:, 2]
                        ),
                        obstacle_position_extractor=lambda states: states.positions(),
                        obstacle_heading_extractor=lambda states: states.headings(),
                    ),
                    distance_threshold=array([0.5], shape=(V,)),
                    weight=10.0,
                    metric=metric,
                ),
            )
            for metric in (
                risk.mean_variance(gamma=2.0, sample_count=50),
                risk.var(alpha=0.95, sample_count=50),
                risk.cvar(alpha=0.95, sample_count=50),
                risk.entropic_risk(theta=0.5, sample_count=50),
            )
        ]

    @mark.parametrize(
        ["inputs", "states", "create_cost"],
        [
            *cases(
                data=data.numpy,
                costs=costs.numpy,
                distance_measure=distance_measure.numpy,
                obstacles=obstacles.numpy,
                risk=risk.numpy,
                types=types.numpy,
            ),
            *cases(
                data=data.jax,
                costs=costs.jax,
                distance_measure=distance_measure.jax,
                obstacles=obstacles.jax,
                risk=risk.jax,
                types=types.jax,
            ),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
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


class test_that_boundary_cost_is_zero_when_boundary_is_far:
    @staticmethod
    def cases(data, costs) -> Sequence[tuple]:
        return [
            (  # Infinitely far boundary
                cost := costs.safety.boundary(
                    distance=stubs.BoundaryDistanceExtractor.returns(
                        data.boundary_distance(np.full((T := 5, M := 4), np.inf)),
                        when_states_are=(
                            states := data.state_batch(
                                np.random.uniform(size=(T, D_x := 2, M)),
                            )
                        ),
                    ),
                    distance_threshold=10.0,
                    weight=5.0,
                ),
                inputs := data.control_input_batch(np.zeros((T, D_u := 2, M))),
                states,
            ),
            (  # Far boundary
                cost := costs.safety.boundary(
                    distance=stubs.BoundaryDistanceExtractor.returns(
                        data.boundary_distance(np.full((T := 5, M := 4), 10.0)),
                        when_states_are=(
                            states := data.state_batch(
                                np.random.uniform(size=(T, D_x := 2, M)),
                            )
                        ),
                    ),
                    distance_threshold=3.0,
                    weight=5.0,
                ),
                inputs := data.control_input_batch(np.zeros((T, D_u := 2, M))),
                states,
            ),
        ]

    @mark.parametrize(
        ["cost", "inputs", "states"],
        [
            *cases(data=data.numpy, costs=costs.numpy),
            *cases(data=data.jax, costs=costs.jax),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
        cost: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
        inputs: ControlInputBatchT,
        states: StateBatchT,
    ) -> None:
        J = np.asarray(cost(inputs=inputs, states=states))

        assert np.allclose(J, 0.0, atol=1e-6), (
            f"Boundary cost should be zero when boundary is infinitely far. Got costs: {J}"
        )


class test_that_boundary_cost_increases_with_decreasing_distance:
    @staticmethod
    def cases(data, costs) -> Sequence[tuple]:
        return [
            (
                cost := costs.safety.boundary(
                    distance=stubs.BoundaryDistanceExtractor.returns(
                        data.boundary_distance(
                            array(
                                [
                                    [0.1, 0.5, 1.0, 5.0],
                                    [0.2, 0.6, 1.5, 6.0],
                                    [0.3, 0.7, 2.0, 7.0],
                                ],
                                shape=(T := 3, M := 4),
                            )
                        ),
                        when_states_are=(
                            states := data.state_batch(
                                np.random.uniform(size=(T, D_x := 2, M)),
                            )
                        ),
                    ),
                    distance_threshold=6.5,  # Only one value is above threshold
                    weight=5.0,
                ),
                inputs := data.control_input_batch(np.zeros((T, D_u := 2, M))),
                states,
                cost_order := [
                    (0, 0),
                    (1, 0),
                    (2, 0),
                    (0, 1),
                    (1, 1),
                    (2, 1),
                    (0, 2),
                    (1, 2),
                    (2, 2),
                    (0, 3),
                    (1, 3),
                    (2, 3),
                ],
            ),
        ]

    @mark.parametrize(
        ["cost", "inputs", "states", "cost_order"],
        [
            *cases(data=data.numpy, costs=costs.numpy),
            *cases(data=data.jax, costs=costs.jax),
        ],
    )
    def test[ControlInputBatchT, StateBatchT, CostsT: Costs](
        self,
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
        ), (
            f"Boundary cost should increase with decreasing distance to boundary. Got costs: {J}"
        )
