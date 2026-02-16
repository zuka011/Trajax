from typing import Sequence

from trajax import ObstacleStateEstimator, model

from numtypes import array

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
