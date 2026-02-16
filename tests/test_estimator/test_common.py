from typing import Sequence, Literal

from trajax import ObstacleStateEstimator, EstimatedObstacleStates, model

from numtypes import Array, array

import numpy as np

from tests.dsl import (
    HasObstacleCount,
    ComponentExtractor,
    ArrayConvertible,
    mppi as data,
    model as model_data,
)
from pytest import mark, Subtests


class test_that_estimated_states_satisfy_basic_properties:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        cases = []

        def random_2d_poses(T: int, K: int):
            return data.obstacle_2d_poses(
                x=array(np.random.randn(T, K), shape=(T, K)),
                y=array(np.random.randn(T, K), shape=(T, K)),
                heading=array(np.random.randn(T, K), shape=(T, K)),
            )

        def random_simple_obstacle_states(T: int, D_o: int, K: int):
            return data.simple_obstacle_states(
                states=array(np.random.randn(T, D_o, K), shape=(T, D_o, K)),
            )

        for T, K in [(1, 1), (2, 5), (4, 3)]:
            cases.extend(
                [
                    (
                        estimator := model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=1.0
                        ),
                        history := random_2d_poses(T, K),
                        expected_count := K,
                    ),
                    (
                        estimator := model.bicycle.estimator.ekf(
                            time_step_size=dt,
                            wheelbase=1.0,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                        history := random_2d_poses(T, K),
                        expected_count := K,
                    ),
                    (
                        estimator := model.bicycle.estimator.ukf(
                            time_step_size=dt,
                            wheelbase=1.0,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                        history := random_2d_poses(T, K),
                        expected_count := K,
                    ),
                    (
                        estimator := model.unicycle.estimator.finite_difference(
                            time_step_size=dt
                        ),
                        history := random_2d_poses(T, K),
                        expected_count := K,
                    ),
                    (
                        estimator := model.unicycle.estimator.ekf(
                            time_step_size=dt,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                        history := random_2d_poses(T, K),
                        expected_count := K,
                    ),
                    (
                        estimator := model.unicycle.estimator.ukf(
                            time_step_size=dt,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                        history := random_2d_poses(T, K),
                        expected_count := K,
                    ),
                    (
                        estimator := model.integrator.estimator.finite_difference(
                            time_step_size=dt
                        ),
                        history := random_simple_obstacle_states(T, D_o := 3, K),
                        expected_count := K,
                    ),
                    (
                        estimator := model.integrator.estimator.kf(
                            time_step_size=dt,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                            observation_dimension=(D_o := 5),
                        ),
                        history := random_simple_obstacle_states(T, D_o, K),
                        expected_count := K,
                    ),
                ]
            )

        return cases

    @mark.parametrize(
        ["estimator", "history", "expected_count"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT: HasObstacleCount, InputsT: HasObstacleCount](
        self,
        subtests: Subtests,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        expected_count: int,
    ) -> None:
        result = estimator.estimate_from(history)

        with subtests.test("estimated states have correct obstacle count"):
            assert result.states.count == expected_count

        with subtests.test("estimated inputs have correct obstacle count"):
            assert result.inputs.count == expected_count

        with subtests.test("estimated states are finite"):
            assert np.all(np.isfinite(result.states))

        with subtests.test("estimated inputs are finite"):
            assert np.all(np.isfinite(result.inputs))


class test_that_estimated_poses_are_close_to_final_historical_poses:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 1e-6
        kalman_tolerance = 0.3
        return [
            *[
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array(
                            [[1.0, 2.0], [1.5, 2.5], [2.0, 3.0], [2.5, 3.5]],
                            shape=(T := 4, K := 2),
                        ),
                        y=array(
                            [[0.0, 1.0], [0.5, 1.5], [1.0, 2.0], [1.5, 2.5]],
                            shape=(T, K),
                        ),
                        heading=array(
                            [[0.0, 0.5], [0.0, 0.75], [0.0, 0.75], [0.0, 1.25]],
                            shape=(T, K),
                        ),
                    ),
                    x_of := lambda result: result.states.x(),
                    y_of := lambda result: result.states.y(),
                    heading_of := lambda result: result.states.heading(),
                    expected_x := array([2.5, 3.5], shape=(K,)),
                    expected_y := array([1.5, 2.5], shape=(K,)),
                    expected_heading := array([0.0, 1.25], shape=(K,)),
                    tolerance,
                )
                for tolerance, estimator in [
                    (
                        tolerance,
                        model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=1.0
                        ),
                    ),
                    (
                        kalman_tolerance,
                        model.bicycle.estimator.ekf(
                            time_step_size=dt,
                            wheelbase=1.0,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                    (
                        kalman_tolerance,
                        model.bicycle.estimator.ukf(
                            time_step_size=dt,
                            wheelbase=1.0,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                    (
                        tolerance,
                        model.unicycle.estimator.finite_difference(time_step_size=dt),
                    ),
                    (
                        kalman_tolerance,
                        model.unicycle.estimator.ekf(
                            time_step_size=dt,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                    (
                        kalman_tolerance,
                        model.unicycle.estimator.ukf(
                            time_step_size=dt,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                ]
            ],
            *[
                (
                    estimator,
                    history := data.simple_obstacle_states(
                        states=array(
                            [
                                [[1.0, 2.0], [0.0, 1.0], [2.0, 4.0]],
                                [[1.5, 2.5], [0.5, 1.5], [3.0, 3.0]],
                                [[2.0, 3.0], [1.0, 2.0], [4.0, 2.0]],
                                [[2.5, 3.5], [1.5, 2.5], [5.5, 2.5]],
                            ],
                            shape=(T := 4, D_o := 3, K := 2),
                        ),
                    ),
                    x_of := lambda result: result.states.array[0],
                    y_of := lambda result: result.states.array[1],
                    heading_of := lambda result: result.states.array[2],
                    expected_x := array([2.5, 3.5], shape=(K,)),
                    expected_y := array([1.5, 2.5], shape=(K,)),
                    expected_heading := array([5.5, 2.5], shape=(K,)),
                    tolerance,
                )
                for tolerance, estimator in [
                    (
                        tolerance,
                        model.integrator.estimator.finite_difference(time_step_size=dt),
                    ),
                    (
                        kalman_tolerance,
                        model.integrator.estimator.kf(
                            time_step_size=dt,
                            process_noise_covariance=array(
                                [1e-5, 1e-5, 1e-5, 1, 1, 1], shape=(6,)
                            ),
                            observation_noise_covariance=1e-6,
                        ),
                    ),
                ]
            ],
        ]

    @mark.parametrize(
        [
            "estimator",
            "history",
            "x_of",
            "y_of",
            "heading_of",
            "expected_x",
            "expected_y",
            "expected_heading",
            "tolerance",
        ],
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
        heading_of: ComponentExtractor[StatesT, InputsT],
        expected_x: Array,
        expected_y: Array,
        expected_heading: Array,
        tolerance: float,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.allclose(x_of(result), expected_x, atol=tolerance)
        assert np.allclose(y_of(result), expected_y, atol=tolerance)
        assert np.allclose(heading_of(result), expected_heading, atol=tolerance)


class test_that_velocity_is_near_zero_for_stationary_obstacles:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 1e-6
        kalman_tolerance = 0.1
        return [
            *[
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array(
                            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], shape=(T := 3, K := 2)
                        ),
                        y=array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]], shape=(T, K)),
                        heading=array(
                            [[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]], shape=(T, K)
                        ),
                    ),
                    velocity_of := lambda result: result.states.speed(),
                    tolerance,
                )
                for tolerance, estimator in [
                    (
                        tolerance,
                        model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=1.0
                        ),
                    ),
                    (
                        kalman_tolerance,
                        model.bicycle.estimator.ekf(
                            time_step_size=dt,
                            wheelbase=1.0,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                    (
                        kalman_tolerance,
                        model.bicycle.estimator.ukf(
                            time_step_size=dt,
                            wheelbase=1.0,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                ]
            ],
            *[
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array(
                            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], shape=(T := 3, K := 2)
                        ),
                        y=array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]], shape=(T, K)),
                        heading=array(
                            [[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]], shape=(T, K)
                        ),
                    ),
                    velocity_of,
                    tolerance,
                )
                for tolerance, estimator in [
                    (
                        tolerance,
                        model.unicycle.estimator.finite_difference(time_step_size=dt),
                    ),
                    (
                        kalman_tolerance,
                        model.unicycle.estimator.ekf(
                            time_step_size=dt,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                    (
                        kalman_tolerance,
                        model.unicycle.estimator.ukf(
                            time_step_size=dt,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                ]
                for velocity_of in [
                    lambda result: result.inputs.linear_velocities(),
                    lambda result: result.inputs.angular_velocities(),
                ]
            ],
            *[
                (
                    estimator,
                    history := data.simple_obstacle_states(
                        states=array(
                            [[[1.0, 2.0], [3.0, 4.0], [0.5, 1.0]]] * 3,
                            shape=(T := 3, D_o := 3, K := 2),
                        ),
                    ),
                    velocity_of := lambda result: np.linalg.norm(result.inputs, axis=0),
                    tolerance,
                )
                for tolerance, estimator in [
                    (
                        tolerance,
                        model.integrator.estimator.finite_difference(time_step_size=dt),
                    ),
                    (
                        kalman_tolerance,
                        model.integrator.estimator.kf(
                            time_step_size=dt,
                            process_noise_covariance=0.1,
                            observation_noise_covariance=array(
                                [1e-5, 2e-5, 3e-5], shape=(3,)
                            ),
                        ),
                    ),
                ]
            ],
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
            *[  # Forward motion for bicycle model
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array([[0.0], [1.0]], shape=(T := 2, K := 1)),
                        y=array([[0.0], [0.0]], shape=(T, K)),
                        heading=array([[0.0], [0.0]], shape=(T, K)),
                    ),
                    velocity_of := lambda result: result.states.speed(),
                    expected_direction := "forward",
                )
                for estimator in [
                    model.bicycle.estimator.finite_difference(
                        time_step_size=dt, wheelbase=1.0
                    ),
                    model.bicycle.estimator.ekf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.bicycle.estimator.ukf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                ]
            ],
            *[  # Backward motion for bicycle model
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array([[0.0], [1.0]], shape=(T := 2, K := 1)),
                        y=array([[0.0], [1.0]], shape=(T, K)),
                        heading=array(
                            [[-np.pi * 3 / 4], [-np.pi * 3 / 4]], shape=(T, K)
                        ),
                    ),
                    velocity_of := lambda result: result.states.speed(),
                    expected_direction := "backward",
                )
                for estimator in [
                    model.bicycle.estimator.finite_difference(
                        time_step_size=dt, wheelbase=1.0
                    ),
                    model.bicycle.estimator.ekf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.bicycle.estimator.ukf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                ]
            ],
            *[  # Forward motion for unicycle model
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array([[0.0], [1.0]], shape=(T := 2, K := 1)),
                        y=array([[0.0], [0.0]], shape=(T, K)),
                        heading=array([[0.0], [0.0]], shape=(T, K)),
                    ),
                    velocity_of := lambda result: result.inputs.linear_velocities(),
                    expected_direction := "forward",
                )
                for estimator in [
                    model.unicycle.estimator.finite_difference(time_step_size=dt),
                    model.unicycle.estimator.ekf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.ukf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                ]
            ],
            *[  # Backward motion for unicycle model
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array([[0.0], [1.0]], shape=(T := 2, K := 1)),
                        y=array([[0.0], [1.0]], shape=(T, K)),
                        heading=array(
                            [[-np.pi * 3 / 4], [-np.pi * 3 / 4]], shape=(T, K)
                        ),
                    ),
                    velocity_of := lambda result: result.inputs.linear_velocities(),
                    expected_direction := "backward",
                )
                for estimator in [
                    model.unicycle.estimator.finite_difference(time_step_size=dt),
                    model.unicycle.estimator.ekf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.ukf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                ]
            ],
            *[  # Forward motion for integrator model
                (
                    estimator,
                    history := data.simple_obstacle_states(
                        states=array(
                            [
                                [[0.0], [0.0], [0.0]],
                                [[1.0], [0.0], [0.0]],
                                [[2.0], [0.0], [0.0]],
                            ],
                            shape=(T := 3, D_o := 3, K := 1),
                        ),
                    ),
                    velocity_of := lambda result: (
                        # Assume [x, y, heading].
                        np.linalg.norm(result.inputs.array[:2], axis=0)
                        * np.cos(result.inputs.array[-1])
                    ),
                    expected_direction := "forward",
                )
                for estimator in [
                    model.integrator.estimator.finite_difference(time_step_size=dt),
                    model.integrator.estimator.kf(
                        time_step_size=dt,
                        process_noise_covariance=0.1,
                        observation_noise_covariance=0.1,
                        observation_dimension=3,
                    ),
                ]
            ],
            *[  # Backward motion for integrator model
                (
                    estimator,
                    history := data.simple_obstacle_states(
                        states=array(
                            [
                                [[0.0], [0.0], [np.pi]],
                                [[-1.0], [0.0], [np.pi]],
                                [[-2.0], [0.0], [np.pi]],
                            ],
                            shape=(T := 3, D_o := 3, K := 1),
                        ),
                    ),
                    velocity_of := lambda result: (
                        # Assume [x, y, heading] in state and [v_x, v_y, angular_velocity] in input.
                        np.linalg.norm(result.inputs.array[:2], axis=0)
                        * np.cos(result.states.array[-1])
                    ),
                    expected_direction := "backward",
                )
                for estimator in [
                    model.integrator.estimator.finite_difference(time_step_size=dt),
                    model.integrator.estimator.kf(
                        time_step_size=dt,
                        process_noise_covariance=0.1,
                        observation_noise_covariance=1e-3,
                        observation_dimension=3,
                    ),
                ]
            ],
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
                assert np.all(velocity_of(result) > 0.0)
            case "backward":
                assert np.all(velocity_of(result) < 0.0)
            case _:
                assert False, f"Unexpected direction: {expected_direction}"


class test_that_estimates_for_constant_velocity_motion_are_close_to_true_velocity:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        T = 10
        dt = 0.1
        velocity = 5.0
        tolerance = 1e-6
        kalman_tolerance = 0.5
        return [
            *[
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array(
                            [[velocity * dt * t] for t in range(T)],
                            shape=(T, K := 1),
                        ),
                        y=array([[0.0] for _ in range(T)], shape=(T, K)),
                        heading=array([[0.0] for _ in range(T)], shape=(T, K)),
                    ),
                    velocity_of := lambda result: result.states.speed(),
                    expected_velocity := velocity,
                    tolerance,
                )
                for tolerance, estimator in [
                    (
                        tolerance,
                        model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=1.0
                        ),
                    ),
                    (
                        kalman_tolerance,
                        model.bicycle.estimator.ekf(
                            time_step_size=dt,
                            wheelbase=1.0,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                    (
                        kalman_tolerance,
                        model.bicycle.estimator.ukf(
                            time_step_size=dt,
                            wheelbase=1.0,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                ]
            ],
            *[
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array(
                            [[velocity * dt * t] for t in range(T)], shape=(T, K := 1)
                        ),
                        y=array([[0.0] for _ in range(T)], shape=(T, K)),
                        heading=array([[0.0] for _ in range(T)], shape=(T, K)),
                    ),
                    velocity_of := lambda result: result.inputs.linear_velocities(),
                    expected_velocity := velocity,
                    tolerance,
                )
                for tolerance, estimator in [
                    (
                        tolerance,
                        model.unicycle.estimator.finite_difference(time_step_size=dt),
                    ),
                    (
                        kalman_tolerance,
                        model.unicycle.estimator.ekf(
                            time_step_size=dt,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                    (
                        kalman_tolerance,
                        model.unicycle.estimator.ukf(
                            time_step_size=dt,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                        ),
                    ),
                ]
            ],
            *[
                (
                    estimator,
                    history := data.simple_obstacle_states(
                        states=array(
                            [[[velocity * dt * t], [0.0]] for t in range(T)],
                            shape=(T, D_o := 2, K := 1),
                        ),
                    ),
                    velocity_of := lambda result: np.linalg.norm(result.inputs, axis=0),
                    expected_velocity := velocity,
                    tolerance,
                )
                for tolerance, estimator in [
                    (
                        tolerance,
                        model.integrator.estimator.finite_difference(time_step_size=dt),
                    ),
                    (
                        kalman_tolerance,
                        model.integrator.estimator.kf(
                            time_step_size=dt,
                            process_noise_covariance=1e-5,
                            observation_noise_covariance=1e-5,
                            observation_dimension=2,
                        ),
                    ),
                ]
            ],
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
            *[
                (
                    estimator,
                    base_history := data.obstacle_2d_poses(
                        x=array([[0.0, 10.0], [1.0, 10.0]], shape=(T := 2, K := 2)),
                        y=array([[0.0, 0.0], [0.0, 0.0]], shape=(T, K)),
                        heading=array([[0.0, 0.0], [0.0, 0.0]], shape=(T, K)),
                    ),
                    perturbed_history := data.obstacle_2d_poses(
                        x=array([[0.0, 15.0], [1.0, 20.0]], shape=(T, K)),
                        y=array([[0.0, 5.0], [0.0, 10.0]], shape=(T, K)),
                        heading=array([[0.0, 0.5], [0.0, 1.0]], shape=(T, K)),
                    ),
                    unperturbed_obstacle_index := 0,
                    tolerance,
                )
                for estimator in [
                    model.bicycle.estimator.finite_difference(
                        time_step_size=dt, wheelbase=1.0
                    ),
                    model.bicycle.estimator.ekf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.bicycle.estimator.ukf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.finite_difference(time_step_size=dt),
                    model.unicycle.estimator.ekf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.ukf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                ]
            ],
            *[
                (
                    estimator,
                    base_history := data.simple_obstacle_states(
                        states=array(
                            [[[0.0, 10.0], [0.0, 0.0]], [[1.0, 10.0], [0.0, 0.0]]],
                            shape=(T := 2, K := 2, D_o := 2),
                        ),
                    ),
                    perturbed_history := data.simple_obstacle_states(
                        states=array(
                            [[[0.0, 15.0], [0.0, 5.0]], [[1.0, 20.0], [0.0, 10.0]]],
                            shape=(T, K, D_o),
                        ),
                    ),
                    unperturbed_obstacle_index := 0,
                    tolerance,
                )
                for estimator in [
                    model.integrator.estimator.finite_difference(time_step_size=dt),
                    model.integrator.estimator.kf(
                        time_step_size=dt,
                        process_noise_covariance=0.1,
                        observation_noise_covariance=1e-3,
                        observation_dimension=2,
                    ),
                ]
            ],
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


class test_that_bicycle_input_estimates_are_close_to_true_values_when_inputs_are_constant:
    @staticmethod
    def cases(model, data, model_data) -> Sequence[tuple]:
        T = 50
        dt = 0.1
        tolerance = 0.1
        kalman_tolerance = 0.2
        cases = []

        def trajectory_for(*, wheelbase: float, acceleration: float, steering: float):
            return model.bicycle.dynamical(
                time_step_size=dt, wheelbase=wheelbase
            ).simulate(
                inputs=model_data.bicycle.control_input_batch(
                    time_horizon=T,
                    rollout_count=1,
                    acceleration=acceleration,
                    steering=steering,
                ),
                initial_state=model_data.bicycle.state(
                    x=15.0, y=12.0, heading=0.5, speed=2.0
                ),
            )

        def acceleration_of(result: EstimatedObstacleStates):
            return result.inputs.accelerations()

        def steering_angle_of(result: EstimatedObstacleStates):
            return result.inputs.steering_angles()

        for wheelbase, acceleration, steering in [
            (1.0, 0.5, 0.15),
            (2.0, -0.5, -0.1),
            (1.5, 0.0, 0.2),
        ]:
            trajectory = trajectory_for(
                wheelbase=wheelbase, acceleration=acceleration, steering=steering
            )
            history = data.obstacle_2d_poses(
                x=array(trajectory.positions.x(), shape=(T, 1)),
                y=array(trajectory.positions.y(), shape=(T, 1)),
                heading=array(trajectory.heading(), shape=(T, 1)),
            )

            cases.extend(
                [
                    (
                        estimator,
                        history,
                        acceleration_of,
                        steering_angle_of,
                        expected_acceleration := acceleration,
                        expected_steering := steering,
                        tolerance,
                    )
                    for tolerance, estimator in [
                        (
                            tolerance,
                            model.bicycle.estimator.finite_difference(
                                time_step_size=dt, wheelbase=wheelbase
                            ),
                        ),
                        (
                            kalman_tolerance,
                            model.bicycle.estimator.ekf(
                                time_step_size=dt,
                                wheelbase=wheelbase,
                                process_noise_covariance=1e-6,
                                observation_noise_covariance=1e-3,
                            ),
                        ),
                        (
                            kalman_tolerance,
                            model.bicycle.estimator.ukf(
                                time_step_size=dt,
                                wheelbase=wheelbase,
                                process_noise_covariance=1e-6,
                                observation_noise_covariance=1e-3,
                            ),
                        ),
                    ]
                ]
            )

        return cases

    @mark.parametrize(
        [
            "estimator",
            "history",
            "acceleration_of",
            "steering_angle_of",
            "expected_acceleration",
            "expected_steering",
            "tolerance",
        ],
        [
            *cases(model=model.numpy, data=data.numpy, model_data=model_data.numpy),
            *cases(model=model.jax, data=data.jax, model_data=model_data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        acceleration_of: ComponentExtractor[StatesT, InputsT],
        steering_angle_of: ComponentExtractor[StatesT, InputsT],
        expected_acceleration: float,
        expected_steering: float,
        tolerance: float,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.allclose(
            acceleration_of(result), expected_acceleration, rtol=tolerance, atol=0.05
        )
        assert np.allclose(
            steering_angle_of(result), expected_steering, rtol=tolerance, atol=0.05
        )


class test_that_unicycle_input_estimates_are_close_to_true_values_when_inputs_are_constant:
    @staticmethod
    def cases(model, data, model_data) -> Sequence[tuple]:
        T = 50
        dt = 0.1
        tolerance = 0.1
        kalman_tolerance = 0.2
        cases = []

        def trajectory_for(*, linear_velocity: float, angular_velocity: float):
            return model.unicycle.dynamical(time_step_size=dt).simulate(
                inputs=model_data.unicycle.control_input_batch(
                    time_horizon=T,
                    rollout_count=1,
                    linear_velocity=linear_velocity,
                    angular_velocity=angular_velocity,
                ),
                initial_state=model_data.unicycle.state(x=15.0, y=12.0, heading=0.5),
            )

        def linear_velocity_of(result: EstimatedObstacleStates):
            return result.inputs.linear_velocities()

        def angular_velocity_of(result: EstimatedObstacleStates):
            return result.inputs.angular_velocities()

        for linear_velocity, angular_velocity in [
            (1.0, 0.5),
            (2.0, -0.25),
            (-5.0, 0.75),
            (0.0, 0.25),
        ]:
            trajectory = trajectory_for(
                linear_velocity=linear_velocity, angular_velocity=angular_velocity
            )
            history = data.obstacle_2d_poses(
                x=array(trajectory.positions.x(), shape=(T, 1)),
                y=array(trajectory.positions.y(), shape=(T, 1)),
                heading=array(trajectory.heading(), shape=(T, 1)),
            )

            cases.extend(
                [
                    (
                        estimator,
                        history,
                        linear_velocity_of,
                        angular_velocity_of,
                        expected_linear := linear_velocity,
                        expected_angular := angular_velocity,
                        tolerance,
                    )
                    for tolerance, estimator in [
                        (
                            tolerance,
                            model.unicycle.estimator.finite_difference(
                                time_step_size=dt
                            ),
                        ),
                        (
                            kalman_tolerance,
                            model.unicycle.estimator.ekf(
                                time_step_size=dt,
                                process_noise_covariance=1e-6,
                                observation_noise_covariance=1e-3,
                            ),
                        ),
                        (
                            kalman_tolerance,
                            model.unicycle.estimator.ukf(
                                time_step_size=dt,
                                process_noise_covariance=1e-6,
                                observation_noise_covariance=1e-3,
                            ),
                        ),
                    ]
                ]
            )

        return cases

    @mark.parametrize(
        [
            "estimator",
            "history",
            "linear_velocity_of",
            "angular_velocity_of",
            "expected_linear",
            "expected_angular",
            "tolerance",
        ],
        [
            *cases(model=model.numpy, data=data.numpy, model_data=model_data.numpy),
            *cases(model=model.jax, data=data.jax, model_data=model_data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        linear_velocity_of: ComponentExtractor[StatesT, InputsT],
        angular_velocity_of: ComponentExtractor[StatesT, InputsT],
        expected_linear: float,
        expected_angular: float,
        tolerance: float,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.allclose(
            linear_velocity_of(result), expected_linear, rtol=tolerance, atol=0.05
        )
        assert np.allclose(
            angular_velocity_of(result), expected_angular, rtol=tolerance, atol=0.05
        )


class test_that_estimates_are_missing_when_obstacle_is_missing:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        T = 3
        K = 3
        missing_obstacle_index = 1

        return [
            *[
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array(
                            [
                                [0.0, 0.0, 10.0],
                                [1.0, np.nan, 10.0],
                                [2.0, np.nan, 10.0],
                            ],
                            shape=(T, K),
                        ),
                        y=array(
                            [
                                [0.0, np.nan, 10.0],
                                [0.0, 0.0, 10.0],
                                [0.0, np.nan, 10.0],
                            ],
                            shape=(T, K),
                        ),
                        heading=array(
                            [
                                [0.0, np.nan, 0.5],
                                [0.0, np.nan, 0.5],
                                [0.0, 0.0, 0.5],
                            ],
                            shape=(T, K),
                        ),
                    ),
                    missing_obstacle_index,
                )
                for estimator in [
                    model.bicycle.estimator.finite_difference(
                        time_step_size=dt, wheelbase=1.0
                    ),
                    model.bicycle.estimator.ekf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.bicycle.estimator.ukf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.finite_difference(time_step_size=dt),
                    model.unicycle.estimator.ekf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.ukf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                ]
            ],
            *[
                (
                    estimator,
                    history := data.simple_obstacle_states(
                        states=array(
                            [
                                [
                                    [0.0, 0.0, 10.0],
                                    [0.0, np.nan, 10.0],
                                    [0.0, np.nan, 10.0],
                                ],
                                [
                                    [1.0, np.nan, 10.0],
                                    [0.0, 0.0, 10.0],
                                    [0.0, np.nan, 10.0],
                                ],
                                [
                                    [2.0, np.nan, 10.0],
                                    [0.0, np.nan, 10.0],
                                    [0.0, 0, 10.0],
                                ],
                            ],
                            shape=(T, D_o := 3, K),
                        ),
                    ),
                    missing_obstacle_index,
                )
                for estimator in [
                    model.integrator.estimator.finite_difference(time_step_size=dt),
                    model.integrator.estimator.kf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                        observation_dimension=3,
                    ),
                ]
            ],
        ]

    @mark.parametrize(
        ["estimator", "history", "missing_obstacle_index"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        subtests: Subtests,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        missing_obstacle_index: int,
    ) -> None:
        result = estimator.estimate_from(history)
        states = np.asarray(result.states)
        inputs = np.asarray(result.inputs)
        covariance = (
            np.asarray(result.covariance) if result.covariance is not None else None
        )

        missing = np.arange(states.shape[-1]) == missing_obstacle_index
        present = np.arange(states.shape[-1]) != missing_obstacle_index

        with subtests.test("missing obstacle estimates are NaN"):
            assert np.all(np.isnan(states[..., missing]))
            assert np.all(np.isnan(inputs[..., missing]))
            assert covariance is None or np.all(np.isnan(covariance[..., missing]))

        with subtests.test("present obstacles have finite states"):
            assert np.all(np.isfinite(states[..., present]))
            assert np.all(np.isfinite(inputs[..., present]))
            assert covariance is None or np.all(np.isfinite(covariance[..., present]))
