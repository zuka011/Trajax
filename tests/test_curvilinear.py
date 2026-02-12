from typing import Sequence

from trajax import (
    model,
    propagator,
    predictor as create_predictor,
    ObstacleStates,
    ObstacleMotionPredictor,
)

from numtypes import array

import numpy as np

from tests.dsl import mppi as data, prediction_creator
from pytest import mark


class test_that_obstacle_motion_is_predicted_correctly:
    @staticmethod
    def cases(create_predictor, model, prediction_creator, data) -> Sequence[tuple]:
        return [
            *[  # Single Integrator CL model tests
                (  # No history
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 5),
                        model=model.integrator.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.integrator(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=np.empty((T_h := 0, K := 0)),
                        y=np.empty((T_h, K)),
                        heading=np.empty((T_h, K)),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=np.empty((T_p, K)),
                        y=np.empty((T_p, K)),
                        heading=np.empty((T_p, K)),
                    ),
                ),
                (  # Single time step history, expected to stay still
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 5),
                        model=model.integrator.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.integrator(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[x := -5.0]], shape=(T_h := 1, K := 1)),
                        y=array([[y := 2.0]], shape=(T_h, K)),
                        heading=array([[theta := 0.0]], shape=(T_h, K)),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=np.full((T_p, K), x),
                        y=np.full((T_p, K), y),
                        heading=np.full((T_p, K), theta),
                    ),
                ),
                (
                    # Multiple time steps, constant velocity
                    # Only last two time steps used for velocity calculation
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.integrator.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.integrator(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array(
                            [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
                            shape=(T_h := 3, K := 2),
                        ),
                        y=array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], shape=(T_h, K)),
                        heading=array(
                            [[0.0, np.pi / 2], [0.0, np.pi / 4], [0.0, 0.0]],
                            shape=(T_h, K),
                        ),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
                            shape=(T_p, K),  # type: ignore
                        ),
                        y=array(
                            [[0.0, 4.0], [0.0, 5.0], [0.0, 6.0], [0.0, 7.0]],
                            shape=(T_p, K),
                        ),
                        heading=array(
                            [
                                [0.0, -np.pi / 4],
                                [0.0, -np.pi / 2],
                                [0.0, -3 * np.pi / 4],
                                [0.0, -np.pi],
                            ],
                            shape=(T_p, K),
                        ),
                    ),
                ),
                (
                    # Multiple time steps, stationary obstacle (in last two steps)
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.integrator.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.integrator(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array(
                            [[-0.5, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
                            shape=(T_h := 3, K := 2),
                        ),
                        y=array([[0.0, 1.0], [0.0, 2.0], [0.0, 2.0]], shape=(T_h, K)),
                        heading=array(
                            [[0.0, np.pi / 2], [0.0, np.pi / 4], [0.0, np.pi / 4]],
                            shape=(T_h, K),
                        ),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
                            shape=(T_p, K),
                        ),
                        y=array(
                            [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]],
                            shape=(T_p, K),
                        ),
                        heading=array(
                            [
                                [0.0, np.pi / 4],
                                [0.0, np.pi / 4],
                                [0.0, np.pi / 4],
                                [0.0, np.pi / 4],
                            ],
                            shape=(T_p, K),
                        ),
                    ),
                ),
            ],
            *[  # Bicycle CL model tests
                (  # No history
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 5),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=1.0
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=np.empty((T_h := 0, K := 0)),
                        y=np.empty((T_h, K)),
                        heading=np.empty((T_h, K)),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=np.empty((T_p, K)),
                        y=np.empty((T_p, K)),
                        heading=np.empty((T_p, K)),
                    ),
                ),
                (  # Single state, zero velocity - stays still
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 5),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=1.0
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[x := 3.0]], shape=(T_h := 1, K := 1)),
                        y=array([[y := 2.0]], shape=(T_h, K)),
                        heading=array([[theta := np.pi / 4]], shape=(T_h, K)),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=np.full((T_p, K), x),
                        y=np.full((T_p, K), y),
                        heading=np.full((T_p, K), theta),
                    ),
                ),
                (  # Single state, moving along x-axis (θ=0)
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=1.0
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[-1.0], [4.0]], shape=(T_h := 2, K := 1)),
                        y=array([[2.0], [2.0]], shape=(T_h, K)),
                        heading=array([[0.0], [0.0]], shape=(T_h, K)),
                    ),
                    # x increases by 5.0 per step, y stays constant
                    expected := data.obstacle_2d_poses(
                        x=array([[9.0], [14.0], [19.0], [24.0]], shape=(T_p, K)),
                        y=np.full((T_p, K), 2.0),
                        heading=np.full((T_p, K), 0.0),
                    ),
                ),
                (  # Moving along y-axis (θ=π/2)
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=1.0
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                        y=array([[0.0], [5.0]], shape=(T_h, K)),
                        heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
                    ),
                    # y increases by 5.0 per step, x stays constant
                    expected := data.obstacle_2d_poses(
                        x=np.full((T_p, K), 0.0),
                        y=array([[10.0], [15.0], [20.0], [25.0]], shape=(T_p, K)),
                        heading=np.full((T_p, K), np.pi / 2),
                    ),
                ),
                (  # Multiple time steps, but obstacle is stationary
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=1.0
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                        y=array([[1.0], [1.0]], shape=(T_h, K)),
                        heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
                    ),
                    # y increases by 5.0 per step, x stays constant
                    expected := data.obstacle_2d_poses(
                        x=np.full((T_p, K), 0.0),
                        y=np.full((T_p, K), 1.0),
                        heading=np.full((T_p, K), np.pi / 2),
                    ),
                ),
                (  # Multiple obstacles with different velocities and headings
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 3),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=1.0
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # Obstacle 0 - v = 10 m/s, θ=0 (moving +x)
                        # Obstacle 1 - v = 10 m/s, θ=π/2 (moving +y)
                        # Obstacle 2 - v = 20 m/s, θ=π (moving -x)
                        x=array(
                            [[0.0, 5.0, 10.0], [1.0, 5.0, 8.0]],
                            shape=(T_h := 2, K := 3),
                        ),
                        y=array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], shape=(T_h, K)),
                        heading=array(
                            [[0.0, np.pi / 2, np.pi], [0.0, np.pi / 2, np.pi]],
                            shape=(T_h, K),
                        ),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [[2.0, 5.0, 6.0], [3.0, 5.0, 4.0], [4.0, 5.0, 2.0]],
                            shape=(T_p, K),
                        ),
                        y=array(
                            [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]],
                            shape=(T_p, K),
                        ),
                        heading=array([[0.0, np.pi / 2, np.pi]] * T_p, shape=(T_p, K)),
                    ),
                ),
                (  # Turning vehicle - constant steering angle (δ) preserved
                    # θ̇ = (v/L) tan(δ), so constant δ means constant angular velocity ω
                    # From history: estimate v and ω, then δ = arctan(ω * L / v)
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # v = 10 m/s (Δ pos = 1.0 per step along heading)
                        # ω = 0.5 rad/s (Δ θ = 0.05 rad per step)
                        # This implies δ = arctan(ω * L / v) = arctan(0.5 * 1 / 10) = arctan(0.05)
                        x=array([[0.0], [1.0]], shape=(T_h := 2, K := 1)),
                        y=array([[0.0], [0.0]], shape=(T_h, K)),
                        heading=array([[0.0], [0.05]], shape=(T_h, K)),  # ω * dt = 0.05
                    ),
                    # Prediction: θ increases by 0.05 each step, path curves
                    # θ(t) = 0.05 * (t + 1) for t = 0, 1, 2, 3
                    # x(t+1) = x(t) + v * cos(θ(t)) * dt
                    # y(t+1) = y(t) + v * sin(θ(t)) * dt
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [
                                [1.0 + 10 * np.cos(0.05) * 0.1],
                                [
                                    1.0
                                    + 10 * np.cos(0.05) * 0.1
                                    + 10 * np.cos(0.10) * 0.1
                                ],
                                [
                                    1.0
                                    + 10 * np.cos(0.05) * 0.1
                                    + 10 * np.cos(0.10) * 0.1
                                    + 10 * np.cos(0.15) * 0.1
                                ],
                                [
                                    1.0
                                    + 10 * np.cos(0.05) * 0.1
                                    + 10 * np.cos(0.10) * 0.1
                                    + 10 * np.cos(0.15) * 0.1
                                    + 10 * np.cos(0.20) * 0.1
                                ],
                            ],
                            shape=(T_p, K),
                        ),
                        y=array(
                            [
                                [10 * np.sin(0.05) * 0.1],
                                [10 * np.sin(0.05) * 0.1 + 10 * np.sin(0.10) * 0.1],
                                [
                                    10 * np.sin(0.05) * 0.1
                                    + 10 * np.sin(0.10) * 0.1
                                    + 10 * np.sin(0.15) * 0.1
                                ],
                                [
                                    10 * np.sin(0.05) * 0.1
                                    + 10 * np.sin(0.10) * 0.1
                                    + 10 * np.sin(0.15) * 0.1
                                    + 10 * np.sin(0.20) * 0.1
                                ],
                            ],
                            shape=(T_p, K),
                        ),
                        heading=array(
                            [[0.10], [0.15], [0.20], [0.25]],
                            shape=(T_p, K),
                        ),
                    ),
                ),
            ],
            *[  # Unicycle CL model tests
                (  # No history
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 5),
                        model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.unicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=np.empty((T_h := 0, K := 0)),
                        y=np.empty((T_h, K)),
                        heading=np.empty((T_h, K)),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=np.empty((T_p, K)),
                        y=np.empty((T_p, K)),
                        heading=np.empty((T_p, K)),
                    ),
                ),
                (  # Single state, zero velocity - stays still
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 5),
                        model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.unicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[x := 3.0]], shape=(T_h := 1, K := 1)),
                        y=array([[y := 2.0]], shape=(T_h, K)),
                        heading=array([[theta := np.pi / 4]], shape=(T_h, K)),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=np.full((T_p, K), x),
                        y=np.full((T_p, K), y),
                        heading=np.full((T_p, K), theta),
                    ),
                ),
                (  # Straight line motion along x-axis (θ=0, ω=0)
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.unicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[-1.0], [4.0]], shape=(T_h := 2, K := 1)),
                        y=array([[2.0], [2.0]], shape=(T_h, K)),
                        heading=array([[0.0], [0.0]], shape=(T_h, K)),
                    ),
                    # v = 50 m/s, ω = 0; x increases by 5.0 per step
                    expected := data.obstacle_2d_poses(
                        x=array([[9.0], [14.0], [19.0], [24.0]], shape=(T_p, K)),
                        y=np.full((T_p, K), 2.0),
                        heading=np.full((T_p, K), 0.0),
                    ),
                ),
                (  # Stationary obstacle (v=0, ω=0)
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.unicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                        y=array([[1.0], [1.0]], shape=(T_h, K)),
                        heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=np.full((T_p, K), 0.0),
                        y=np.full((T_p, K), 1.0),
                        heading=np.full((T_p, K), np.pi / 2),
                    ),
                ),
                (  # Multiple obstacles with different velocities and headings
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 3),
                        model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.unicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # Obstacle 0 - v = 10 m/s, θ=0 (moving +x)
                        # Obstacle 1 - v = 10 m/s, θ=π/2 (moving +y)
                        # Obstacle 2 - v = 20 m/s, θ=π (moving -x)
                        x=array(
                            [[0.0, 5.0, 10.0], [1.0, 5.0, 8.0]],
                            shape=(T_h := 2, K := 3),
                        ),
                        y=array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], shape=(T_h, K)),
                        heading=array(
                            [[0.0, np.pi / 2, np.pi], [0.0, np.pi / 2, np.pi]],
                            shape=(T_h, K),
                        ),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [[2.0, 5.0, 6.0], [3.0, 5.0, 4.0], [4.0, 5.0, 2.0]],
                            shape=(T_p, K),
                        ),
                        y=array(
                            [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]],
                            shape=(T_p, K),
                        ),
                        heading=array([[0.0, np.pi / 2, np.pi]] * T_p, shape=(T_p, K)),
                    ),
                ),
                (  # Turning motion with constant ω
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.unicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # v = 10 m/s (Δ pos = 1.0 per step along heading)
                        # ω = 0.5 rad/s (Δ θ = 0.05 rad per step)
                        x=array([[x_0 := 0.0], [x_1 := 2.0]], shape=(T_h := 2, K := 1)),
                        y=array([[0.0], [0.0]], shape=(T_h, K)),
                        # ω * dt = 0.05
                        heading=array(
                            [[theta_0 := 0.0], [theta_1 := 0.05]], shape=(T_h, K)
                        ),
                    ),
                    # Prediction: θ increases by 0.05 each step, path curves
                    # Same as bicycle for this case since both predict constant ω
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [
                                [
                                    x_1
                                    + (v := x_1 - x_0)
                                    * np.cos((w := theta_1 - theta_0) * 1)
                                ],
                                [x_1 + v * np.cos(w * 1) + v * np.cos(w * 2)],
                                [
                                    x_1
                                    + v * np.cos(w * 1)
                                    + v * np.cos(w * 2)
                                    + v * np.cos(w * 3)
                                ],
                                [
                                    x_1
                                    + v * np.cos(w * 1)
                                    + v * np.cos(w * 2)
                                    + v * np.cos(w * 3)
                                    + v * np.cos(w * 4)
                                ],
                            ],
                            shape=(T_p, K),
                        ),
                        y=array(
                            [
                                [v * np.sin(w * 1)],
                                [v * np.sin(w * 1) + v * np.sin(w * 2)],
                                [
                                    v * np.sin(w * 1)
                                    + v * np.sin(w * 2)
                                    + v * np.sin(w * 3)
                                ],
                                [
                                    v * np.sin(w * 1)
                                    + v * np.sin(w * 2)
                                    + v * np.sin(w * 3)
                                    + v * np.sin(w * 4)
                                ],
                            ],
                            shape=(T_p, K),
                        ),
                        heading=array(
                            [
                                [theta_1 + w * 1],
                                [theta_1 + w * 2],
                                [theta_1 + w * 3],
                                [theta_1 + w * 4],
                            ],
                            shape=(T_p, K),
                        ),
                    ),
                ),
                (  # Pure rotation (v=0, ω≠0) - spinning in place
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                        prediction=prediction_creator.unicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # v = 0 (stationary position), ω = 1.0 rad/s
                        x=array([[5.0], [5.0]], shape=(T_h := 2, K := 1)),
                        y=array([[3.0], [3.0]], shape=(T_h, K)),
                        heading=array([[0.0], [0.1]], shape=(T_h, K)),  # ω * dt = 0.1
                    ),
                    # Position stays constant, only heading changes
                    expected := data.obstacle_2d_poses(
                        x=np.full((T_p, K), 5.0),
                        y=np.full((T_p, K), 3.0),
                        heading=array(
                            [[0.2], [0.3], [0.4], [0.5]],
                            shape=(T_p, K),
                        ),
                    ),
                ),
            ],
        ]

    @mark.parametrize(
        ["predictor", "history", "expected"],
        [
            *cases(
                create_predictor=create_predictor.numpy,
                model=model.numpy,
                prediction_creator=prediction_creator.numpy,
                data=data.numpy,
            ),
            *cases(
                create_predictor=create_predictor.jax,
                model=model.jax,
                prediction_creator=prediction_creator.jax,
                data=data.jax,
            ),
        ],
    )
    def test[HistoryT, PredictionT: ObstacleStates](
        self,
        predictor: ObstacleMotionPredictor[HistoryT, PredictionT],
        history: HistoryT,
        expected: PredictionT,
    ) -> None:
        actual = predictor.predict(history=history)
        assert np.allclose(actual.x(), expected.x(), rtol=1e-3, atol=1e-6)
        assert np.allclose(actual.y(), expected.y(), rtol=1e-3, atol=1e-6)
        assert np.allclose(actual.heading(), expected.heading(), rtol=1e-3, atol=1e-6)


class test_that_no_covariance_information_is_provided_when_propagator_is_not_available:
    @staticmethod
    def cases(create_predictor, model, data, prediction_creator) -> Sequence[tuple]:
        return [
            (
                predictor := create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=1.0
                    ),
                    prediction=prediction_creator.bicycle(),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                    y=array([[1.0], [1.0]], shape=(T_h, K)),
                    heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
                ),
            ),
        ]

    @mark.parametrize(
        ["predictor", "history"],
        [
            *cases(
                create_predictor=create_predictor.numpy,
                model=model.numpy,
                data=data.numpy,
                prediction_creator=prediction_creator.numpy,
            ),
            *cases(
                create_predictor=create_predictor.jax,
                model=model.jax,
                data=data.jax,
                prediction_creator=prediction_creator.jax,
            ),
        ],
    )
    def test[HistoryT, PredictionT: ObstacleStates](
        self,
        predictor: ObstacleMotionPredictor[HistoryT, PredictionT],
        history: HistoryT,
    ) -> None:
        assert predictor.predict(history=history).covariance() is None


class test_that_position_covariance_information_is_provided_when_propagator_is_available:
    @staticmethod
    def cases(
        create_predictor, model, data, propagator, prediction_creator
    ) -> Sequence[tuple]:
        return [
            (
                predictor := create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=1.0
                    ),
                    propagator=propagator.linear(
                        time_step_size=dt,
                        # TODO: Review!
                        covariance=propagator.covariance.composite(
                            state_provider=propagator.covariance.constant_variance(
                                variance=0.1, dimension=2
                            ),
                            input_provider=propagator.covariance.constant_variance(
                                variance=0.2, dimension=2
                            ),
                        ),
                        resizing=propagator.covariance.resize(pad_to=3, epsilon=1e-15),
                    ),
                    prediction=prediction_creator.bicycle(),
                ),
                prediction_horizon := T_p,
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                    y=array([[1.0], [1.0]], shape=(T_h, K)),
                    heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
                ),
            ),
        ]

    @mark.parametrize(
        ["predictor", "prediction_horizon", "history"],
        [
            *cases(
                create_predictor=create_predictor.numpy,
                model=model.numpy,
                data=data.numpy,
                propagator=propagator.numpy,
                prediction_creator=prediction_creator.numpy,
            ),
            *cases(
                create_predictor=create_predictor.jax,
                model=model.jax,
                data=data.jax,
                propagator=propagator.jax,
                prediction_creator=prediction_creator.jax,
            ),
        ],
    )
    def test[HistoryT, PredictionT: ObstacleStates](
        self,
        predictor: ObstacleMotionPredictor[HistoryT, PredictionT],
        prediction_horizon: int,
        history: HistoryT,
    ) -> None:
        covariances = np.asarray(predictor.predict(history=history).covariance())

        assert np.all(
            [
                (covariances[t + 1, 0, 0] > covariances[t, 0, 0])
                & (covariances[t + 1, 1, 1] > covariances[t, 1, 1])
                for t in range(prediction_horizon - 1)
            ]
        )


class test_that_velocity_assumptions_are_applied_during_prediction:
    @staticmethod
    def cases(create_predictor, model, data, prediction_creator) -> Sequence[tuple]:
        return [
            (  # Integrator: zero out the heading velocity (index 2), keep x and y
                create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.integrator.obstacle(time_step_size=(dt := 0.1)),
                    prediction=prediction_creator.integrator(),
                    assumptions=lambda velocity: velocity.zeroed(at=(2,)),
                ),
                data.obstacle_2d_poses(
                    x=array([[0.0, 0.0], [1.0, 0.0]], shape=(T_h := 2, K := 2)),
                    y=array([[0.0, -1.0], [1.0, 1.0]], shape=(T_h, K)),
                    heading=array([[0.0, np.pi / 2], [0.0, np.pi]], shape=(T_h, K)),
                ),
                data.obstacle_2d_poses(
                    x=array(
                        [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]], shape=(T_p, K)
                    ),
                    y=array(
                        [[2.0, 3.0], [3.0, 5.0], [4.0, 7.0], [5.0, 9.0]], shape=(T_p, K)
                    ),
                    # Heading stays constant.
                    heading=array([[0.0, np.pi]] * T_p, shape=(T_p, K)),
                ),
            ),
            (  # Bicycle: zero out steering angle, obstacle goes straight
                create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                    ),
                    prediction=prediction_creator.bicycle(),
                    assumptions=lambda velocity: velocity.zeroed(steering_angle=True),
                ),
                data.obstacle_2d_poses(
                    # v = 10 m/s, ω = 0.5 rad/s => δ = arctan(ω * L / v)
                    # With assumption: δ = 0, so heading stays constant
                    x=array([[0.0], [1.0]], shape=(T_h := 2, K := 1)),
                    y=array([[0.0], [0.0]], shape=(T_h, K)),
                    heading=array([[0.0], [0.05]], shape=(T_h, K)),
                ),
                # With zeroed steering: straight-line motion at heading=0.05, speed=10
                data.obstacle_2d_poses(
                    x=array(
                        [
                            [1.0 + 10 * np.cos(0.05) * 0.1],
                            [1.0 + 10 * np.cos(0.05) * 0.1 * 2],
                            [1.0 + 10 * np.cos(0.05) * 0.1 * 3],
                            [1.0 + 10 * np.cos(0.05) * 0.1 * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    y=array(
                        [
                            [10 * np.sin(0.05) * 0.1],
                            [10 * np.sin(0.05) * 0.1 * 2],
                            [10 * np.sin(0.05) * 0.1 * 3],
                            [10 * np.sin(0.05) * 0.1 * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    heading=array([[0.05]] * T_p, shape=(T_p, K)),
                ),
            ),
            (  # Unicycle: zero out angular velocity, obstacle goes straight
                create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                    prediction=prediction_creator.unicycle(),
                    assumptions=lambda velocity: velocity.zeroed(angular_velocity=True),
                ),
                data.obstacle_2d_poses(
                    # v = 10 m/s, ω = 0.5 rad/s
                    # With assumption: ω = 0, straight-line motion
                    x=array([[0.0], [1.0]], shape=(T_h := 2, K := 1)),
                    y=array([[0.0], [0.0]], shape=(T_h, K)),
                    heading=array([[0.0], [0.05]], shape=(T_h, K)),
                ),
                # With zeroed angular velocity: straight-line at heading=0.05, v=10
                data.obstacle_2d_poses(
                    x=array(
                        [
                            [1.0 + 10 * np.cos(0.05) * 0.1],
                            [1.0 + 10 * np.cos(0.05) * 0.1 * 2],
                            [1.0 + 10 * np.cos(0.05) * 0.1 * 3],
                            [1.0 + 10 * np.cos(0.05) * 0.1 * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    y=array(
                        [
                            [10 * np.sin(0.05) * 0.1],
                            [10 * np.sin(0.05) * 0.1 * 2],
                            [10 * np.sin(0.05) * 0.1 * 3],
                            [10 * np.sin(0.05) * 0.1 * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    heading=array([[0.05]] * T_p, shape=(T_p, K)),
                ),
            ),
        ]

    @mark.parametrize(
        ["predictor", "history", "expected"],
        [
            *cases(
                create_predictor=create_predictor.numpy,
                model=model.numpy,
                data=data.numpy,
                prediction_creator=prediction_creator.numpy,
            ),
            *cases(
                create_predictor=create_predictor.jax,
                model=model.jax,
                data=data.jax,
                prediction_creator=prediction_creator.jax,
            ),
        ],
    )
    def test[HistoryT, PredictionT: ObstacleStates](
        self,
        predictor: ObstacleMotionPredictor[HistoryT, PredictionT],
        history: HistoryT,
        expected: PredictionT,
    ) -> None:
        actual = predictor.predict(history=history)
        assert np.allclose(actual.x(), expected.x(), rtol=1e-3, atol=1e-6)
        assert np.allclose(actual.y(), expected.y(), rtol=1e-3, atol=1e-6)
        assert np.allclose(actual.heading(), expected.heading(), rtol=1e-3, atol=1e-6)
