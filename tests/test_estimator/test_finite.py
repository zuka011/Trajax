from typing import Sequence

from trajax import ObstacleStateEstimator, model

from numtypes import array

import numpy as np

from tests.dsl import ComponentExtractor, mppi as data
from pytest import mark


class test_that_velocity_estimates_are_zero_for_single_state_history:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 1e-10
        return [
            (
                estimator := model.bicycle.estimator.finite_difference(
                    time_step_size=dt, wheelbase=1.0
                ),
                history := data.obstacle_2d_poses(
                    x=array([[5.0, 10.0]], shape=(1, 2)),
                    y=array([[3.0, 7.0]], shape=(1, 2)),
                    heading=array([[0.5, 1.0]], shape=(1, 2)),
                ),
                velocity_of := lambda result: result.states.speed(),
                tolerance,
            ),
            (
                estimator := model.unicycle.estimator.finite_difference(
                    time_step_size=dt
                ),
                history := data.obstacle_2d_poses(
                    x=array([[5.0, 10.0]], shape=(1, 2)),
                    y=array([[3.0, 7.0]], shape=(1, 2)),
                    heading=array([[0.5, 1.0]], shape=(1, 2)),
                ),
                velocity_of := lambda result: result.inputs.linear_velocities(),
                tolerance,
            ),
            (
                estimator := model.integrator.estimator.finite_difference(
                    time_step_size=dt
                ),
                history := data.simple_obstacle_states(
                    states=array([[[5.0, 10.0], [3.0, 7.0]]], shape=(1, 2, 2)),
                ),
                velocity_of := lambda result: result.inputs.array,
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


class test_that_acceleration_is_zero_for_fewer_than_three_states:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        tolerance = 1e-10
        return [
            (
                estimator := model.bicycle.estimator.finite_difference(
                    time_step_size=dt, wheelbase=1.0
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.0]], shape=(1, 1)),
                    y=array([[0.0]], shape=(1, 1)),
                    heading=array([[0.0]], shape=(1, 1)),
                ),
                acceleration_of := lambda result: result.inputs.accelerations(),
                tolerance,
            ),
            (
                estimator := model.bicycle.estimator.finite_difference(
                    time_step_size=dt, wheelbase=1.0
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.0]], shape=(2, 1)),
                    heading=array([[0.0], [0.0]], shape=(2, 1)),
                ),
                acceleration_of := lambda result: result.inputs.accelerations(),
                tolerance,
            ),
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
