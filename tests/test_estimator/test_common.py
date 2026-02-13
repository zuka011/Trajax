from typing import Sequence, Literal

from trajax import ObstacleStateEstimator, model

from numtypes import Array, array

import numpy as np

from tests.dsl import (
    ArrayConvertible,
    HasObstacleCount,
    ComponentExtractor,
    mppi as data,
)
from pytest import mark


class test_that_obstacle_count_in_estimated_states_matches_historical_obstacle_count:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        return [
            *[
                (
                    estimator := model.bicycle.estimator(
                        time_step_size=dt, wheelbase=1.0
                    ),
                    history := data.obstacle_2d_poses(
                        x=array(np.random.randn(T, K), shape=(T, K)),
                        y=array(np.random.randn(T, K), shape=(T, K)),
                        heading=array(np.random.randn(T, K), shape=(T, K)),
                    ),
                    expected_count := K,
                )
                for T, K in [(1, 1), (3, 2), (5, 5), (2, 10)]
            ],
            *[
                (
                    estimator := model.unicycle.estimator(time_step_size=dt),
                    history := data.obstacle_2d_poses(
                        x=array(np.random.randn(T, K), shape=(T, K)),
                        y=array(np.random.randn(T, K), shape=(T, K)),
                        heading=array(np.random.randn(T, K), shape=(T, K)),
                    ),
                    expected_count := K,
                )
                for T, K in [(1, 1), (3, 2), (5, 5), (2, 10)]
            ],
            *[
                (
                    estimator := model.integrator.estimator(time_step_size=dt),
                    history := data.simple_obstacle_states(
                        states=array(np.random.randn(T, D_o, K), shape=(T, D_o, K)),
                    ),
                    expected_count := K,
                )
                for T, D_o, K in [(1, 3, 1), (3, 3, 2), (5, 2, 5), (2, 4, 10)]
            ],
        ]

    @mark.parametrize(
        ["estimator", "history", "expected_count"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT: HasObstacleCount, InputsT: HasObstacleCount](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        expected_count: int,
    ) -> None:
        result = estimator.estimate_from(history)
        assert result.states.count == expected_count


class test_that_estimated_positions_are_close_to_final_historical_positions:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        return [
            *[
                (
                    estimator := model.bicycle.estimator(
                        time_step_size=dt, wheelbase=1.0
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]], shape=(3, 2)),
                        y=array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]], shape=(3, 2)),
                        heading=array(
                            [[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]], shape=(3, 2)
                        ),
                    ),
                    x_of := lambda result: result.states.x(),
                    y_of := lambda result: result.states.y(),
                    expected_x := array([2.0, 3.0], shape=(2,)),
                    expected_y := array([1.0, 2.0], shape=(2,)),
                ),
            ],
            *[
                (
                    estimator := model.unicycle.estimator(time_step_size=dt),
                    history := data.obstacle_2d_poses(
                        x=array([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]], shape=(3, 2)),
                        y=array([[0.0, 1.0], [0.5, 1.5], [1.0, 2.0]], shape=(3, 2)),
                        heading=array(
                            [[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]], shape=(3, 2)
                        ),
                    ),
                    x_of := lambda result: result.states.x(),
                    y_of := lambda result: result.states.y(),
                    expected_x := array([2.0, 3.0], shape=(2,)),
                    expected_y := array([1.0, 2.0], shape=(2,)),
                ),
            ],
            *[
                (
                    estimator := model.integrator.estimator(time_step_size=dt),
                    history := data.simple_obstacle_states(
                        states=array(
                            [
                                [[1.0, 2.0], [0.0, 1.0]],
                                [[1.5, 2.5], [0.5, 1.5]],
                                [[2.0, 3.0], [1.0, 2.0]],
                            ],
                            shape=(3, 2, 2),
                        ),
                    ),
                    x_of := lambda result: result.states.array[0],
                    y_of := lambda result: result.states.array[1],
                    expected_x := array([2.0, 3.0], shape=(2,)),
                    expected_y := array([1.0, 2.0], shape=(2,)),
                ),
            ],
        ]

    @mark.parametrize(
        ["estimator", "history", "x_of", "y_of", "expected_x", "expected_y"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        x_of: ComponentExtractor[StatesT, InputsT],
        y_of: ComponentExtractor[StatesT, InputsT],
        expected_x: Array,
        expected_y: Array,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.allclose(x_of(result), expected_x, atol=1e-6)
        assert np.allclose(y_of(result), expected_y, atol=1e-6)


class test_that_estimated_heading_is_close_to_final_historical_heading:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        return [
            *[
                (
                    estimator := model.bicycle.estimator(
                        time_step_size=dt, wheelbase=1.0
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[1.0, 2.0], [1.5, 2.5]], shape=(2, 2)),
                        y=array([[0.0, 1.0], [0.5, 1.5]], shape=(2, 2)),
                        heading=array([[0.1, 0.3], [0.2, 0.4]], shape=(2, 2)),
                    ),
                    heading_of := lambda result: result.states.heading(),
                    expected_heading := array([0.2, 0.4], shape=(2,)),
                ),
            ],
            *[
                (
                    estimator := model.unicycle.estimator(time_step_size=dt),
                    history := data.obstacle_2d_poses(
                        x=array([[1.0, 2.0], [1.5, 2.5]], shape=(2, 2)),
                        y=array([[0.0, 1.0], [0.5, 1.5]], shape=(2, 2)),
                        heading=array([[0.1, 0.3], [0.2, 0.4]], shape=(2, 2)),
                    ),
                    heading_of := lambda result: result.states.heading(),
                    expected_heading := array([0.2, 0.4], shape=(2,)),
                ),
            ],
        ]

    @mark.parametrize(
        ["estimator", "history", "heading_of", "expected_heading"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        heading_of: ComponentExtractor[StatesT, InputsT],
        expected_heading: Array,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.allclose(heading_of(result), expected_heading, atol=1e-6)


class test_that_all_estimates_are_finite:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        return [
            *[
                (
                    estimator := model.bicycle.estimator(
                        time_step_size=dt, wheelbase=1.0
                    ),
                    history := data.obstacle_2d_poses(
                        x=array(np.random.randn(T, K), shape=(T, K)),
                        y=array(np.random.randn(T, K), shape=(T, K)),
                        heading=array(np.random.randn(T, K), shape=(T, K)),
                    ),
                )
                for T, K in [(1, 1), (2, 2), (3, 3), (5, 5)]
            ],
            *[
                (
                    estimator := model.unicycle.estimator(time_step_size=dt),
                    history := data.obstacle_2d_poses(
                        x=array(np.random.randn(T, K), shape=(T, K)),
                        y=array(np.random.randn(T, K), shape=(T, K)),
                        heading=array(np.random.randn(T, K), shape=(T, K)),
                    ),
                )
                for T, K in [(1, 1), (2, 2), (3, 3), (5, 5)]
            ],
            *[
                (
                    estimator := model.integrator.estimator(time_step_size=dt),
                    history := data.simple_obstacle_states(
                        states=array(
                            np.random.randn(T, D_o, K) * 10, shape=(T, D_o, K)
                        ),
                    ),
                )
                for T, D_o, K in [(1, 3, 1), (2, 3, 2), (3, 2, 3), (5, 4, 5)]
            ],
        ]

    @mark.parametrize(
        ["estimator", "history"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT: ArrayConvertible, InputsT: ArrayConvertible](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.all(np.isfinite(result.states))
        assert np.all(np.isfinite(result.inputs))


class test_that_velocity_is_near_zero_for_stationary_obstacles:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 1e-6
        return [
            (
                estimator := model.bicycle.estimator(time_step_size=dt, wheelbase=1.0),
                history := data.obstacle_2d_poses(
                    x=array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], shape=(3, 2)),
                    y=array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]], shape=(3, 2)),
                    heading=array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]], shape=(3, 2)),
                ),
                velocity_of := lambda result: result.states.speed(),
                tolerance,
            ),
            (
                estimator := model.unicycle.estimator(time_step_size=dt),
                history := data.obstacle_2d_poses(
                    x=array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], shape=(3, 2)),
                    y=array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]], shape=(3, 2)),
                    heading=array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]], shape=(3, 2)),
                ),
                velocity_of := lambda result: result.inputs.linear_velocities(),
                tolerance,
            ),
            (
                estimator := model.integrator.estimator(time_step_size=dt),
                history := data.simple_obstacle_states(
                    states=array(
                        [[[1.0, 2.0], [3.0, 4.0], [0.5, 1.0]]] * 3, shape=(3, 3, 2)
                    ),
                ),
                velocity_of := lambda result: np.linalg.norm(result.inputs, axis=0),
                tolerance,
            ),
        ]

    @mark.parametrize(
        ["estimator", "history", "velocity_of", "tolerance"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        velocity_of: ComponentExtractor[StatesT, InputsT],
        tolerance: float,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.allclose(velocity_of(result), 0.0, atol=tolerance)


class test_that_velocity_direction_matches_motion_direction:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        return [
            (  # Forward motion for bicycle model
                estimator := model.bicycle.estimator(time_step_size=dt, wheelbase=1.0),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.0]], shape=(2, 1)),
                    heading=array([[0.0], [0.0]], shape=(2, 1)),
                ),
                velocity_of := lambda result: result.states.speed(),
                expected_direction := "forward",
            ),
            (  # Backward motion for bicycle model
                estimator := model.bicycle.estimator(time_step_size=dt, wheelbase=1.0),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [1.0]], shape=(2, 1)),
                    heading=array([[-np.pi * 3 / 4], [-np.pi * 3 / 4]], shape=(2, 1)),
                ),
                velocity_of := lambda result: result.states.speed(),
                expected_direction := "backward",
            ),
            (  # Forward motion for unicycle model
                estimator := model.unicycle.estimator(time_step_size=dt),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.0]], shape=(2, 1)),
                    heading=array([[0.0], [0.0]], shape=(2, 1)),
                ),
                velocity_of := lambda result: result.inputs.linear_velocities(),
                expected_direction := "forward",
            ),
            (  # Backward motion for unicycle model
                estimator := model.unicycle.estimator(time_step_size=dt),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [1.0]], shape=(2, 1)),
                    heading=array([[-np.pi * 3 / 4], [-np.pi * 3 / 4]], shape=(2, 1)),
                ),
                velocity_of := lambda result: result.inputs.linear_velocities(),
                expected_direction := "backward",
            ),
        ]

    @mark.parametrize(
        ["estimator", "history", "velocity_of", "expected_direction"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        velocity_of: ComponentExtractor[StatesT, InputsT],
        expected_direction: Literal["forward", "backward"],
    ) -> None:
        result = estimator.estimate_from(history)

        match expected_direction:
            case "forward":
                assert np.all(velocity_of(result) >= 0.0)
            case "backward":
                assert np.all(velocity_of(result) <= 0.0)
            case _:
                assert False, f"Unexpected direction: {expected_direction}"


class test_that_estimates_for_constant_velocity_motion_are_close_to_true_velocity:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        velocity = 5.0
        tolerance = 1e-6
        return [
            (
                estimator := model.bicycle.estimator(time_step_size=dt, wheelbase=1.0),
                history := data.obstacle_2d_poses(
                    x=array(
                        [[0.0], [velocity * dt], [velocity * dt * 2]], shape=(3, 1)
                    ),
                    y=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                    heading=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                ),
                velocity_of := lambda result: result.states.speed(),
                expected_velocity := velocity,
                tolerance,
            ),
            (
                estimator := model.unicycle.estimator(time_step_size=dt),
                history := data.obstacle_2d_poses(
                    x=array(
                        [[0.0], [velocity * dt], [velocity * dt * 2]], shape=(3, 1)
                    ),
                    y=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                    heading=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                ),
                velocity_of := lambda result: result.inputs.linear_velocities(),
                expected_velocity := velocity,
                tolerance,
            ),
            (
                estimator := model.integrator.estimator(time_step_size=dt),
                history := data.simple_obstacle_states(
                    states=array(
                        [
                            [[0.0], [0.0]],
                            [[velocity * dt], [0.0]],
                            [[velocity * dt * 2], [0.0]],
                        ],
                        shape=(3, 2, 1),
                    ),
                ),
                velocity_of := lambda result: np.linalg.norm(result.inputs, axis=0),
                expected_velocity := velocity,
                tolerance,
            ),
        ]

    @mark.parametrize(
        ["estimator", "history", "velocity_of", "expected_velocity", "tolerance"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        velocity_of: ComponentExtractor[StatesT, InputsT],
        expected_velocity: float,
        tolerance: float,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.allclose(velocity_of(result), expected_velocity, atol=tolerance)


class test_that_estimates_are_independent_across_obstacles:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 1e-6
        return [
            (
                estimator := model.bicycle.estimator(time_step_size=dt, wheelbase=1.0),
                base_history := data.obstacle_2d_poses(
                    x=array([[0.0, 10.0], [1.0, 10.0]], shape=(2, 2)),
                    y=array([[0.0, 0.0], [0.0, 0.0]], shape=(2, 2)),
                    heading=array([[0.0, 0.0], [0.0, 0.0]], shape=(2, 2)),
                ),
                perturbed_history := data.obstacle_2d_poses(
                    x=array([[0.0, 15.0], [1.0, 20.0]], shape=(2, 2)),
                    y=array([[0.0, 5.0], [0.0, 10.0]], shape=(2, 2)),
                    heading=array([[0.0, 0.5], [0.0, 1.0]], shape=(2, 2)),
                ),
                unperturbed_obstacle_index := 0,
                tolerance,
            ),
            (
                estimator := model.unicycle.estimator(time_step_size=dt),
                base_history := data.obstacle_2d_poses(
                    x=array([[0.0, 10.0], [1.0, 10.0]], shape=(2, 2)),
                    y=array([[0.0, 0.0], [0.0, 0.0]], shape=(2, 2)),
                    heading=array([[0.0, 0.0], [0.0, 0.0]], shape=(2, 2)),
                ),
                perturbed_history := data.obstacle_2d_poses(
                    x=array([[0.0, 15.0], [1.0, 20.0]], shape=(2, 2)),
                    y=array([[0.0, 5.0], [0.0, 10.0]], shape=(2, 2)),
                    heading=array([[0.0, 0.5], [0.0, 1.0]], shape=(2, 2)),
                ),
                unperturbed_obstacle_index := 0,
                tolerance,
            ),
            (
                estimator := model.integrator.estimator(time_step_size=dt),
                base_history := data.simple_obstacle_states(
                    states=array(
                        [[[0.0, 10.0], [0.0, 0.0]], [[1.0, 10.0], [0.0, 0.0]]],
                        shape=(2, 2, 2),
                    ),
                ),
                perturbed_history := data.simple_obstacle_states(
                    states=array(
                        [[[0.0, 15.0], [0.0, 5.0]], [[1.0, 20.0], [0.0, 10.0]]],
                        shape=(2, 2, 2),
                    ),
                ),
                unperturbed_obstacle_index := 0,
                tolerance,
            ),
        ]

    @mark.parametrize(
        [
            "estimator",
            "base_history",
            "perturbed_history",
            "unperturbed_obstacle_index",
            "tolerance",
        ],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT: ArrayConvertible, InputsT: ArrayConvertible](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        base_history: HistoryT,
        perturbed_history: HistoryT,
        unperturbed_obstacle_index: int,
        tolerance: float,
    ) -> None:
        base = estimator.estimate_from(base_history)
        perturbed = estimator.estimate_from(perturbed_history)

        base_states = np.asarray(base.states.array)[:, unperturbed_obstacle_index]
        perturbed_states = np.asarray(perturbed.states.array)[
            :, unperturbed_obstacle_index
        ]
        base_inputs = np.asarray(base.inputs)[:, unperturbed_obstacle_index]
        perturbed_inputs = np.asarray(perturbed.inputs)[:, unperturbed_obstacle_index]

        assert np.allclose(base_states, perturbed_states, atol=tolerance)
        assert np.allclose(base_inputs, perturbed_inputs, atol=tolerance)


class test_that_steering_is_zero_for_constant_heading:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 1e-6
        return [
            (
                estimator := model.bicycle.estimator(time_step_size=dt, wheelbase=1.0),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0], [2.0]], shape=(3, 1)),
                    y=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                    heading=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                ),
                steering_angle_of := lambda result: result.inputs.steering_angles(),
                tolerance,
            ),
            (
                estimator := model.bicycle.estimator(time_step_size=dt, wheelbase=2.0),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0], [2.0]], shape=(3, 1)),
                    y=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                    heading=array([[0.5], [0.5], [0.5]], shape=(3, 1)),
                ),
                steering_angle_of := lambda result: result.inputs.steering_angles(),
                tolerance,
            ),
        ]

    @mark.parametrize(
        ["estimator", "history", "steering_angle_of", "tolerance"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        steering_angle_of: ComponentExtractor[StatesT, InputsT],
        tolerance: float,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.allclose(steering_angle_of(result), 0.0, atol=tolerance)


class test_that_angular_velocity_is_zero_for_constant_heading:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 1e-6
        return [
            (
                estimator := model.unicycle.estimator(time_step_size=dt),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.0]], shape=(2, 1)),
                    heading=array([[0.0], [0.0]], shape=(2, 1)),
                ),
                angular_velocity_of := lambda result: (
                    result.inputs.angular_velocities()
                ),
                tolerance,
            ),
            (
                estimator := model.unicycle.estimator(time_step_size=dt),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.0]], shape=(2, 1)),
                    heading=array([[np.pi / 4], [np.pi / 4]], shape=(2, 1)),
                ),
                angular_velocity_of := lambda result: (
                    result.inputs.angular_velocities()
                ),
                tolerance,
            ),
        ]

    @mark.parametrize(
        ["estimator", "history", "angular_velocity_of", "tolerance"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        angular_velocity_of: ComponentExtractor[StatesT, InputsT],
        tolerance: float,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.allclose(angular_velocity_of(result), 0.0, atol=tolerance)


class test_that_steering_scales_with_wheelbase:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        small_wheelbase = 1.0
        large_wheelbase = 2.0
        velocity = 10.0
        delta_heading = 0.1
        return [
            (
                small_estimator := model.bicycle.estimator(
                    time_step_size=dt, wheelbase=small_wheelbase
                ),
                large_estimator := model.bicycle.estimator(
                    time_step_size=dt, wheelbase=large_wheelbase
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [velocity * dt]], shape=(2, 1)),
                    y=array([[0.0], [0.0]], shape=(2, 1)),
                    heading=array([[0.0], [delta_heading]], shape=(2, 1)),
                ),
                steering_angle_of := lambda result: result.inputs.steering_angles(),
            ),
        ]

    @mark.parametrize(
        ["small_estimator", "large_estimator", "history", "steering_angle_of"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        small_estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        large_estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        steering_angle_of: ComponentExtractor[StatesT, InputsT],
    ) -> None:
        small_result = small_estimator.estimate_from(history)
        large_result = large_estimator.estimate_from(history)

        assert np.all(
            np.abs(steering_angle_of(large_result))
            > np.abs(steering_angle_of(small_result))
        )
