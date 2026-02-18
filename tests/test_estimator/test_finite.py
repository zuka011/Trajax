from typing import Sequence, Callable

from trajax import ObstacleStateEstimator, EstimatedObstacleStates, model

from numtypes import Array, array

import numpy as np

from tests.dsl import ComponentExtractor, ArrayConvertible, mppi as data
from pytest import mark


class test_that_velocity_estimates_are_zero_for_single_state_history:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 1e-10
        return [
            *[
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array([[5.0, 10.0]], shape=(T := 1, K := 2)),
                        y=array([[3.0, 7.0]], shape=(T, K)),
                        heading=array([[0.5, 1.0]], shape=(T, K)),
                    ),
                    velocity_of,
                    tolerance,
                )
                for estimator, velocity_of in [
                    (
                        model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=1.0
                        ),
                        [
                            lambda result: result.states.speed(),
                            # Steering angle is kinda like a velocity.
                            lambda result: result.inputs.steering_angles(),
                        ],
                    ),
                    (
                        model.unicycle.estimator.finite_difference(time_step_size=dt),
                        [
                            lambda result: result.inputs.linear_velocities(),
                            lambda result: result.inputs.angular_velocities(),
                        ],
                    ),
                ]
            ],
            (
                estimator := model.integrator.estimator.finite_difference(
                    time_step_size=dt
                ),
                history := data.simple_obstacle_states(
                    states=array(
                        [[[5.0, 10.0], [3.0, 7.0]]], shape=(T := 1, D_o := 2, K := 2)
                    ),
                ),
                velocity_of := [lambda result: result.inputs.array],
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
        velocity_of: Sequence[ComponentExtractor[StatesT, InputsT]],
        tolerance: float,
    ) -> None:
        result = estimator.estimate_from(history)
        for velocity in velocity_of:
            assert np.allclose(velocity(result), 0.0, atol=tolerance)


class test_that_acceleration_is_zero_for_fewer_than_three_states:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 1e-10
        histories = [
            # One historical state.
            data.obstacle_2d_poses(
                x=array([[0.0]], shape=(T := 1, K := 1)),
                y=array([[0.0]], shape=(T, K)),
                heading=array([[0.0]], shape=(T, K)),
            ),
            # Two historical states.
            history := data.obstacle_2d_poses(
                x=array([[0.0], [1.0]], shape=(T := 2, K := 1)),
                y=array([[0.0], [0.0]], shape=(T, K)),
                heading=array([[0.0], [0.0]], shape=(T, K)),
            ),
        ]

        return [
            (
                estimator := model.bicycle.estimator.finite_difference(
                    time_step_size=dt, wheelbase=1.0
                ),
                history,
                acceleration_of := lambda result: result.inputs.accelerations(),
                tolerance,
            )
            for history in histories
        ]

    @mark.parametrize(
        ["estimator", "history", "acceleration_of", "tolerance"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        acceleration_of: ComponentExtractor[StatesT, InputsT],
        tolerance: float,
    ) -> None:
        result = estimator.estimate_from(history)
        assert np.allclose(acceleration_of(result), 0.0, atol=tolerance)


class test_that_finite_difference_estimators_return_none_covariance:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        return [
            *[
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array([[0.0, 1.0], [0.5, 1.5]], shape=(T := 2, K := 2)),
                        y=array([[0.0, 0.0], [0.5, 0.5]], shape=(T, K)),
                        heading=array([[0.7, 0.7], [0.7, 0.7]], shape=(T, K)),
                    ),
                )
                for estimator in [
                    model.bicycle.estimator.finite_difference(
                        time_step_size=dt, wheelbase=1.0
                    ),
                    model.unicycle.estimator.finite_difference(time_step_size=dt),
                ]
            ],
            (
                estimator := model.integrator.estimator.finite_difference(
                    time_step_size=dt
                ),
                history := data.simple_obstacle_states(
                    states=array(
                        [[[0.0, 1.0], [0.0, 0.0]], [[0.5, 1.5], [0.5, 0.5]]],
                        shape=(T := 2, D_o := 2, K := 2),
                    ),
                ),
            ),
        ]

    @mark.parametrize(
        ["estimator", "history"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT, None],
        history: HistoryT,
    ) -> None:
        result = estimator.estimate_from(history)
        assert result.covariance is None


class test_that_missing_state_earlier_in_history_does_not_affect_estimates:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        T = 8
        D_o = 3
        K = 2

        def poses_history(*, missing_at: int | None = None):
            x = np.arange(T * K, dtype=float).reshape(T, K)
            y = np.arange(T * K, dtype=float).reshape(T, K) * 0.5
            heading = np.zeros((T, K))

            if missing_at is not None:
                x[missing_at] = np.nan
                y[missing_at] = np.nan
                heading[missing_at] = np.nan

            return data.obstacle_2d_poses(
                x=array(x, shape=(T, K)),
                y=array(y, shape=(T, K)),
                heading=array(heading, shape=(T, K)),
            )

        def simple_history(*, missing_at: int | None = None):
            states = np.arange(T * K * D_o, dtype=float).reshape(T, D_o, K)
            states[:, 1, :] = states[:, 0, :] * 0.5
            states[:, 2, :] = 0.0

            if missing_at is not None:
                states[missing_at] = np.nan

            return data.simple_obstacle_states(states=array(states, shape=(T, D_o, K)))

        return [
            *[
                (
                    estimator := model.bicycle.estimator.finite_difference(
                        time_step_size=dt, wheelbase=1.0
                    ),
                    history := poses_history(missing_at=t),
                    reference_history := poses_history(),
                )
                for t in range(T - 6, T - 4)
            ],
            *[
                (
                    estimator := model.unicycle.estimator.finite_difference(
                        time_step_size=dt
                    ),
                    history := poses_history(missing_at=t),
                    reference_history := poses_history(),
                )
                for t in range(T - 5, T - 3)
            ],
            *[
                (
                    estimator := model.integrator.estimator.finite_difference(
                        time_step_size=dt
                    ),
                    history := simple_history(missing_at=t),
                    reference_history := simple_history(),
                )
                for t in range(T - 5, T - 3)
            ],
        ]

    @mark.parametrize(
        ["estimator", "history", "reference_history"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT: ArrayConvertible, InputsT: ArrayConvertible](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        reference_history: HistoryT,
    ) -> None:
        result = estimator.estimate_from(history)
        reference_result = estimator.estimate_from(reference_history)

        assert np.allclose(result.states, reference_result.states)
        assert np.allclose(result.inputs, reference_result.inputs)


class test_that_estimable_states_are_computed_when_there_are_not_enough_valid_entries_for_all_states:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 0.1

        def check_exact(
            actual: ArrayConvertible, expected: ArrayConvertible | None
        ) -> bool:
            if expected is None:
                assert np.all(np.isnan(actual))
            else:
                assert np.allclose(actual, expected, rtol=tolerance)

            return True

        def check_in_range(
            actual: ArrayConvertible, expected_range: tuple[float, float] | None
        ) -> bool:
            if expected_range is None:
                assert np.all(np.isnan(actual))
            else:
                low, high = expected_range
                assert np.all((actual > low) & (actual < high))

            return True

        def check_bicycle(
            *,
            expected_x: float | None = None,
            expected_y: float | None = None,
            expected_heading: float | None = None,
            expected_speed: float | None = None,
            expected_steering_angle_range: tuple[float, float] | None = None,
            expected_acceleration_range: tuple[float, float] | None = None,
        ):
            def check(result: EstimatedObstacleStates) -> bool:
                assert check_exact(result.states.x(), expected_x)
                assert check_exact(result.states.y(), expected_y)
                assert check_exact(result.states.heading(), expected_heading)
                assert check_exact(result.states.speed(), expected_speed)
                assert check_in_range(
                    result.inputs.steering_angles(), expected_steering_angle_range
                )
                assert check_in_range(
                    result.inputs.accelerations(), expected_acceleration_range
                )

                return True

            return check

        def check_unicycle(
            *,
            expected_x: float | None = None,
            expected_y: float | None = None,
            expected_heading: float | None = None,
            expected_linear_velocity: float | None = None,
            expected_angular_velocity: float | None = None,
        ):
            def check(result: EstimatedObstacleStates) -> bool:
                assert check_exact(result.states.x(), expected_x)
                assert check_exact(result.states.y(), expected_y)
                assert check_exact(result.states.heading(), expected_heading)
                assert check_exact(
                    result.inputs.linear_velocities(), expected_linear_velocity
                )
                assert check_exact(
                    result.inputs.angular_velocities(), expected_angular_velocity
                )

                return True

            return check

        def check_integrator(
            *,
            expected_states: Array | None = None,
            expected_velocities: Array | None = None,
        ):
            def check(result: EstimatedObstacleStates) -> bool:
                assert check_exact(result.states.array, expected_states)
                assert check_exact(result.inputs.array, expected_velocities)

                return True

            return check

        return [
            *[  # Sufficient states for pose
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array(
                            [[np.nan], [np.nan], [2.0]],
                            shape=(T := 3, K := 1),
                        ),
                        y=array([[np.nan], [np.nan], [0.0]], shape=(T, K)),
                        heading=array(
                            [[np.nan], [np.nan], [np.pi / 16]],
                            shape=(T, K),
                        ),
                    ),
                    check,
                )
                for estimator, check in [
                    (
                        model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=1.0
                        ),
                        check_bicycle(
                            expected_x=2.0,
                            expected_y=0.0,
                            expected_heading=np.pi / 16,
                        ),
                    ),
                    (
                        model.unicycle.estimator.finite_difference(time_step_size=dt),
                        check_unicycle(
                            expected_x=2.0,
                            expected_y=0.0,
                            expected_heading=np.pi / 16,
                        ),
                    ),
                ]
            ],
            *[  # Sufficient states for velocities
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array([[np.nan], [1.0], [2.0]], shape=(T := 3, K := 1)),
                        y=array([[np.nan], [0.0], [0.0]], shape=(T, K)),
                        heading=array([[np.nan], [0.0], [np.pi / 16]], shape=(T, K)),
                    ),
                    check,
                )
                for estimator, check in [
                    (
                        model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=1.0
                        ),
                        check_bicycle(
                            expected_x=2.0,
                            expected_y=0.0,
                            expected_heading=np.pi / 16,
                            expected_speed=10.0,
                            expected_steering_angle_range=(0.0, 2 * np.pi / 16),
                        ),
                    ),
                    (
                        model.unicycle.estimator.finite_difference(time_step_size=dt),
                        check_unicycle(
                            expected_x=2.0,
                            expected_y=0.0,
                            expected_heading=np.pi / 16,
                            expected_linear_velocity=10.0,
                            expected_angular_velocity=np.pi / 16 / dt,
                        ),
                    ),
                ]
            ],
            (  # Sufficient states for acceleration.
                estimator := model.bicycle.estimator.finite_difference(
                    time_step_size=dt, wheelbase=1.0
                ),
                history := data.obstacle_2d_poses(
                    x=array(
                        [[np.nan], [1.0], [2.0], [4.0]],
                        shape=(T := 4, K := 1),
                    ),
                    y=array([[np.nan], [0.0], [0.0], [0.0]], shape=(T, K)),
                    heading=array([[np.nan], [0.0], [0.0], [0.0]], shape=(T, K)),
                ),
                check_bicycle(
                    expected_x=4.0,
                    expected_y=0.0,
                    expected_heading=0.0,
                    expected_speed=20.0,
                    expected_steering_angle_range=(-0.1, 0.1),
                    expected_acceleration_range=(0.0, 150.0),
                ),
            ),
            (  # Sufficient states for integrator states.
                estimator := model.integrator.estimator.finite_difference(
                    time_step_size=dt
                ),
                history := data.simple_obstacle_states(
                    states=array(
                        [
                            [[np.nan], [np.nan]],
                            [[np.nan], [np.nan]],
                            [[np.nan], [np.nan]],
                            [[4.0], [2.0]],
                        ],
                        shape=(T := 4, D_o := 2, K := 1),
                    ),
                ),
                check_integrator(
                    expected_states=array([[4.0], [2.0]], shape=(D_o, K)),
                ),
            ),
            (  # Sufficient states for integrator velocities.
                estimator := model.integrator.estimator.finite_difference(
                    time_step_size=dt
                ),
                history := data.simple_obstacle_states(
                    states=array(
                        [
                            [[np.nan], [np.nan]],
                            [[1.0], [0.5]],
                            [[4.0], [2.0]],
                        ],
                        shape=(T := 3, D_o := 2, K := 1),
                    ),
                ),
                check_integrator(
                    expected_states=array([[4.0], [2.0]], shape=(D_o, K)),
                    expected_velocities=array([[30.0], [15.0]], shape=(D_o, K)),
                ),
            ),
        ]

    @mark.parametrize(
        ["estimator", "history", "check"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        check: Callable[[EstimatedObstacleStates], bool],
    ) -> None:
        result = estimator.estimate_from(history)

        assert check(result)
