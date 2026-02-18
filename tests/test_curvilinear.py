from typing import Sequence

from trajax import (
    model,
    predictor as create_predictor,
    ObstacleStates,
    ObstacleMotionPredictor,
)

from numtypes import array

import numpy as np

from tests.dsl import mppi as data, prediction_creator, compute
from pytest import mark


class test_that_obstacle_motion_is_predicted_correctly:
    @staticmethod
    def cases(create_predictor, model, prediction_creator, data) -> Sequence[tuple]:
        return [
            *[  # Single Integrator CL model tests
                (  # No history
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 5),
                        model=model.integrator.obstacle(
                            time_step_size=(dt := 0.1), state_dimension=3
                        ),
                        estimator=model.integrator.estimator.finite_difference(
                            time_step_size=dt
                        ),
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
                        model=model.integrator.obstacle(
                            time_step_size=(dt := 0.1), state_dimension=3
                        ),
                        estimator=model.integrator.estimator.finite_difference(
                            time_step_size=dt
                        ),
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
                        model=model.integrator.obstacle(
                            time_step_size=(dt := 0.1), state_dimension=3
                        ),
                        estimator=model.integrator.estimator.finite_difference(
                            time_step_size=dt
                        ),
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
                        model=model.integrator.obstacle(
                            time_step_size=(dt := 0.1), state_dimension=3
                        ),
                        estimator=model.integrator.estimator.finite_difference(
                            time_step_size=dt
                        ),
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
                            time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
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
                            time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
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
                            time_step_size=(dt := 0.1),
                            wheelbase=(L := 1.0),
                            process_noise_covariance=0.0,
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
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
                            time_step_size=(dt := 0.1),
                            wheelbase=(L := 1.0),
                            process_noise_covariance=0.0,
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
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
                            time_step_size=(dt := 0.1),
                            wheelbase=(L := 1.0),
                            process_noise_covariance=0.0,
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
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
                            time_step_size=(dt := 0.1),
                            wheelbase=(L := 1.0),
                            process_noise_covariance=0.0,
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
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
                            time_step_size=(dt := 0.1),
                            wheelbase=(L := 1.0),
                            process_noise_covariance=0.0,
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # v ≈ 9.9875 m/s (Δ pos = 1.0 and θ=0.05)
                        # ω = 0.5 rad/s (Δ θ = 0.05 rad per step)
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
                                [
                                    (x_0 := 1.0)
                                    + (v := 1.0 * np.cos(0.05) / dt) * np.cos(0.05) * dt
                                ],
                                [
                                    (x_1 := x_0 + v * np.cos(0.05) * dt)
                                    + v * np.cos(0.10) * dt
                                ],
                                [
                                    (x_2 := x_1 + v * np.cos(0.10) * dt)
                                    + v * np.cos(0.15) * dt
                                ],
                                [
                                    (x_3 := x_2 + v * np.cos(0.15) * dt)
                                    + v * np.cos(0.20) * dt
                                ],
                            ],
                            shape=(T_p, K),
                        ),
                        y=array(
                            [
                                [(y_0 := 0.0) + v * np.sin(0.05) * dt],
                                [
                                    (y_1 := y_0 + v * np.sin(0.05) * dt)
                                    + v * np.sin(0.10) * dt
                                ],
                                [
                                    (y_2 := y_1 + v * np.sin(0.10) * dt)
                                    + v * np.sin(0.15) * dt
                                ],
                                [
                                    (y_3 := y_2 + v * np.sin(0.15) * dt)
                                    + v * np.sin(0.20) * dt
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
                (  # Accelerating vehicle along x-axis (θ=0, δ=0)
                    # History shows increasing speed: v = 8, 10 m/s (a = 20 m/s²)
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1),
                            wheelbase=(L := 1.0),
                            process_noise_covariance=0.0,
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[0.0], [0.8], [x_0 := 1.8]], shape=(T_h := 3, K := 1)),
                        y=array([[0.0], [0.0], [0.0]], shape=(T_h, K)),
                        heading=array([[0.0], [0.0], [0.0]], shape=(T_h, K)),
                    ),
                    # Current state: x=1.8, v=10, a=20, δ=0
                    # Bicycle model: x(t+1) = x(t) + v(t) * dt, v(t+1) = v(t) + a * dt
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [
                                [x_1 := x_0 + (v_0 := 10) * dt],
                                [x_2 := x_1 + (v_1 := v_0 + 2) * dt],
                                [x_3 := x_2 + (v_2 := v_1 + 2) * dt],
                                [x_4 := x_3 + (v_3 := v_2 + 2) * dt],
                            ],
                            shape=(T_p, K),
                        ),
                        y=np.full((T_p, K), 0.0),
                        heading=np.full((T_p, K), 0.0),
                    ),
                ),
                (  # Reverse motion: vehicle moving backward (opposite to heading)
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1),
                            wheelbase=(L := 1.0),
                            process_noise_covariance=0.0,
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                        # Heading is π/2 (pointing up), but vehicle moves down (negative y)
                        y=array([[0.0], [-1.0]], shape=(T_h, K)),
                        heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
                    ),
                    # Speed estimation with projection onto heading:
                    # v = (Δx * cos(θ) + Δy * sin(θ)) / dt
                    # v = (0 * cos(π/2) + (-1) * sin(π/2)) / 0.1 = (-1) / 0.1 = -10 m/s
                    expected := data.obstacle_2d_poses(
                        x=np.full((T_p, K), 0.0),
                        y=array(
                            [
                                [(y_0 := -1.0) + (v := -10.0) * np.sin(np.pi / 2) * dt],
                                [y_0 + v * np.sin(np.pi / 2) * dt * 2],
                                [y_0 + v * np.sin(np.pi / 2) * dt * 3],
                                [y_0 + v * np.sin(np.pi / 2) * dt * 4],
                            ],
                            shape=(T_p, K),
                        ),
                        heading=np.full((T_p, K), np.pi / 2),
                    ),
                ),
                (  # Motion with constant acceleration and steering angle
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1),
                            wheelbase=(L := 1.0),
                            process_noise_covariance=0.0,
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # x increases by 1.0 each step -> v = 10 m/s constant
                        x=array([[0.0], [1.0], [x_0 := 2.0]], shape=(T_h := 3, K := 1)),
                        y=array([[0.0], [0.0], [0.0]], shape=(T_h, K)),
                        heading=array([[0.0], [0.0], [0.0]], shape=(T_h, K)),
                    ),
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [
                                [x_0 + (v := 10.0) * dt * 1],
                                [x_0 + v * dt * 2],
                                [x_0 + v * dt * 3],
                                [x_0 + v * dt * 4],
                            ],
                            shape=(T_p, K),
                        ),
                        y=np.full((T_p, K), 0.0),
                        heading=np.full((T_p, K), 0.0),
                    ),
                ),
                (  # Decelerating vehicle
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1),
                            wheelbase=(L := 1.0),
                            process_noise_covariance=0.0,
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # x: 0 -> 1.2 -> 2.2 (Δx = 1.2, then 1.0)
                        # v_0 = 1.2/0.1 = 12, v_1 = 1.0/0.1 = 10 -> a = -20 m/s²
                        x=array([[0.0], [1.2], [x_0 := 2.2]], shape=(T_h := 3, K := 1)),
                        y=array([[0.0], [0.0], [0.0]], shape=(T_h, K)),
                        heading=array([[0.0], [0.0], [0.0]], shape=(T_h, K)),
                    ),
                    # Current state: x=2.2, v=10, a=-20, δ=0
                    # Vehicle slows down by 2 m/s each step
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [
                                [x_1 := x_0 + (v_0 := 10.0) * dt],
                                [x_2 := x_1 + (v_1 := v_0 - 2) * dt],
                                [x_3 := x_2 + (v_2 := v_1 - 2) * dt],
                                [x_4 := x_3 + (v_3 := v_2 - 2) * dt],
                            ],
                            shape=(T_p, K),
                        ),
                        y=np.full((T_p, K), 0.0),
                        heading=np.full((T_p, K), 0.0),
                    ),
                ),
                (  # Turning while reversing
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1),
                            wheelbase=(L := 1.0),
                            process_noise_covariance=0.0,
                        ),
                        estimator=model.bicycle.estimator.finite_difference(
                            time_step_size=dt, wheelbase=L
                        ),
                        prediction=prediction_creator.bicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # Vehicle pointing right (θ=0) but moving left (reverse)
                        # θ changes from 0.05 to 0.00
                        x=array([[0.0], [-1.0]], shape=(T_h := 2, K := 1)),
                        y=array([[0.0], [0.0]], shape=(T_h, K)),
                        heading=array([[0.05], [0.0]], shape=(T_h, K)),
                    ),
                    # v = (Δx * cos(θ) + Δy * sin(θ)) / dt = (-1 * 1 + 0) / 0.1 = -10 m/s
                    # ω = (0.0 - 0.05) / 0.1 = -0.5 rad/s
                    # δ = arctan(L * ω / v) = arctan(1 * -0.5 / -10) = arctan(0.05) ≈ 0.05
                    # When reversing, negative ω with negative v gives positive steering
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [
                                [
                                    (x_0 := -1.0)
                                    + (v := -10.0) * np.cos((theta_0 := 0.0)) * dt
                                ],
                                [
                                    (x_1 := x_0 + v * np.cos(theta_0) * dt)
                                    + v
                                    * np.cos(theta_1 := theta_0 + (w := -0.5) * dt)
                                    * dt
                                ],
                                [
                                    (x_2 := x_1 + v * np.cos(theta_1) * dt)
                                    + v * np.cos(theta_2 := theta_1 + w * dt) * dt
                                ],
                                [
                                    (x_3 := x_2 + v * np.cos(theta_2) * dt)
                                    + v * np.cos(theta_3 := theta_2 + w * dt) * dt
                                ],
                            ],
                            shape=(T_p, K),
                        ),
                        y=array(
                            [
                                [v * np.sin(theta_0) * dt],
                                [v * np.sin(theta_0) * dt + v * np.sin(theta_1) * dt],
                                [
                                    v * np.sin(theta_0) * dt
                                    + v * np.sin(theta_1) * dt
                                    + v * np.sin(theta_2) * dt
                                ],
                                [
                                    v * np.sin(theta_0) * dt
                                    + v * np.sin(theta_1) * dt
                                    + v * np.sin(theta_2) * dt
                                    + v * np.sin(theta_3) * dt
                                ],
                            ],
                            shape=(T_p, K),
                        ),
                        heading=array(
                            [
                                [theta_0 + w * dt],
                                [theta_0 + w * dt * 2],
                                [theta_0 + w * dt * 3],
                                [theta_0 + w * dt * 4],
                            ],
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
                        estimator=model.unicycle.estimator.finite_difference(
                            time_step_size=dt
                        ),
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
                        estimator=model.unicycle.estimator.finite_difference(
                            time_step_size=dt
                        ),
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
                        estimator=model.unicycle.estimator.finite_difference(
                            time_step_size=dt
                        ),
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
                        estimator=model.unicycle.estimator.finite_difference(
                            time_step_size=dt
                        ),
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
                        estimator=model.unicycle.estimator.finite_difference(
                            time_step_size=dt
                        ),
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
                        estimator=model.unicycle.estimator.finite_difference(
                            time_step_size=dt
                        ),
                        prediction=prediction_creator.unicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # Δ pos = 2.0, θ = 0.05, estimate: v ≈ 19.975 m/s
                        # ω = 0.5 rad/s (Δ θ = 0.05 rad per step)
                        x=array([[x_0 := 0.0], [x_1 := 2.0]], shape=(T_h := 2, K := 1)),
                        y=array([[0.0], [0.0]], shape=(T_h, K)),
                        # ω * dt = 0.05
                        heading=array(
                            [[theta_0 := 0.0], [theta_1 := 0.05]], shape=(T_h, K)
                        ),
                    ),
                    # Prediction: θ increases by 0.05 each step, path curves
                    expected := data.obstacle_2d_poses(
                        x=array(
                            [
                                [
                                    x_1
                                    + (v := (x_1 - x_0) * np.cos(theta_1) / dt)
                                    * np.cos((w := theta_1 - theta_0) * 1)
                                    * dt
                                ],
                                [x_1 + v * np.cos(w * 1) * dt + v * np.cos(w * 2) * dt],
                                [
                                    x_1
                                    + v * np.cos(w * 1) * dt
                                    + v * np.cos(w * 2) * dt
                                    + v * np.cos(w * 3) * dt
                                ],
                                [
                                    x_1
                                    + v * np.cos(w * 1) * dt
                                    + v * np.cos(w * 2) * dt
                                    + v * np.cos(w * 3) * dt
                                    + v * np.cos(w * 4) * dt
                                ],
                            ],
                            shape=(T_p, K),
                        ),
                        y=array(
                            [
                                [v * np.sin(w * 1) * dt],
                                [v * np.sin(w * 1) * dt + v * np.sin(w * 2) * dt],
                                [
                                    v * np.sin(w * 1) * dt
                                    + v * np.sin(w * 2) * dt
                                    + v * np.sin(w * 3) * dt
                                ],
                                [
                                    v * np.sin(w * 1) * dt
                                    + v * np.sin(w * 2) * dt
                                    + v * np.sin(w * 3) * dt
                                    + v * np.sin(w * 4) * dt
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
                        estimator=model.unicycle.estimator.finite_difference(
                            time_step_size=dt
                        ),
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
                (  # Reverse motion
                    predictor := create_predictor.curvilinear(
                        horizon=(T_p := 4),
                        model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                        estimator=model.unicycle.estimator.finite_difference(
                            time_step_size=dt
                        ),
                        prediction=prediction_creator.unicycle(),
                    ),
                    history := data.obstacle_2d_poses(
                        # Heading is π/2 (pointing up), but vehicle moves down (negative y)
                        x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                        y=array([[0.0], [-1.0]], shape=(T_h, K)),
                        heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
                    ),
                    # Speed estimation with projection onto heading:
                    # v = (Δx * cos(θ) + Δy * sin(θ)) / dt
                    # v = (0 * cos(π/2) + (-1) * sin(π/2)) / 0.1 = -10 m/s
                    expected := data.obstacle_2d_poses(
                        x=np.full((T_p, K), 0.0),
                        y=array(
                            [
                                [(y_0 := -1.0) + (v := -10.0) * np.sin(np.pi / 2) * dt],
                                [y_0 + v * np.sin(np.pi / 2) * dt * 2],
                                [y_0 + v * np.sin(np.pi / 2) * dt * 3],
                                [y_0 + v * np.sin(np.pi / 2) * dt * 4],
                            ],
                            shape=(T_p, K),
                        ),
                        heading=np.full((T_p, K), np.pi / 2),
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
        assert np.allclose(actual.x(), expected.x(), rtol=1e-2, atol=1e-3)
        assert np.allclose(actual.y(), expected.y(), rtol=1e-2, atol=1e-3)
        assert np.allclose(actual.heading(), expected.heading(), rtol=1e-2, atol=1e-3)


class test_that_uncertainty_is_small_when_estimator_does_not_provide_it:
    @staticmethod
    def cases(create_predictor, model, data, prediction_creator) -> Sequence[tuple]:
        return [
            (
                predictor := create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1),
                        wheelbase=(L := 1.0),
                        process_noise_covariance=0.0,
                    ),
                    estimator=model.bicycle.estimator.finite_difference(
                        time_step_size=dt, wheelbase=L
                    ),
                    prediction=prediction_creator.bicycle(),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                    y=array([[1.0], [1.0]], shape=(T_h, K)),
                    heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
                ),
            ),
            (
                predictor := create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.unicycle.obstacle(
                        time_step_size=(dt := 0.1), process_noise_covariance=0.0
                    ),
                    estimator=model.unicycle.estimator.finite_difference(
                        time_step_size=dt
                    ),
                    prediction=prediction_creator.unicycle(),
                ),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                    y=array([[1.0], [1.0]], shape=(T_h, K)),
                    heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
                ),
            ),
            (
                predictor := create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.integrator.obstacle(
                        time_step_size=(dt := 0.1),
                        state_dimension=3,
                        process_noise_covariance=0.0,
                    ),
                    estimator=model.integrator.estimator.finite_difference(
                        time_step_size=dt
                    ),
                    prediction=prediction_creator.integrator(),
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
        assert np.allclose(
            predictor.predict(history=history).covariance(), 0.0, atol=1e-2
        )


class test_that_position_covariance_information_is_provided_when_estimator_provides_covariance:
    @staticmethod
    def cases(create_predictor, model, data, prediction_creator) -> Sequence[tuple]:
        return [
            (
                predictor := create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                    ),
                    estimator=model.bicycle.estimator.ekf(
                        time_step_size=dt,
                        wheelbase=L,
                        process_noise_covariance=0.01,
                        observation_noise_covariance=0.01,
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
        ), f"Expected covariance to increase over time steps, but got {covariances}"


class test_that_input_assumptions_are_applied_during_prediction:
    @staticmethod
    def cases(create_predictor, model, data, prediction_creator) -> Sequence[tuple]:
        return [
            (  # Integrator: zero out the heading velocity (index 2), keep x and y
                create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.integrator.obstacle(
                        time_step_size=(dt := 0.1), state_dimension=3
                    ),
                    estimator=model.integrator.estimator.finite_difference(
                        time_step_size=dt
                    ),
                    prediction=prediction_creator.integrator(),
                    assumptions=lambda inputs: inputs.zeroed(at=(2,)),
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
                    estimator=model.bicycle.estimator.finite_difference(
                        time_step_size=dt, wheelbase=L
                    ),
                    prediction=prediction_creator.bicycle(),
                    assumptions=lambda inputs: inputs.zeroed(steering_angle=True),
                ),
                data.obstacle_2d_poses(
                    # v is estimated using projection: (1 * cos(0.05)) / 0.1 ≈ 9.9875
                    # With assumption: δ = 0, so heading stays constant
                    x=array([[0.0], [1.0]], shape=(T_h := 2, K := 1)),
                    y=array([[0.0], [0.0]], shape=(T_h, K)),
                    heading=array([[0.0], [0.05]], shape=(T_h, K)),
                ),
                # With zeroed steering: straight-line motion at heading=0.05
                data.obstacle_2d_poses(
                    x=array(
                        [
                            [1.0 + (v := 1.0 * np.cos(0.05) / dt) * np.cos(0.05) * dt],
                            [1.0 + v * np.cos(0.05) * dt * 2],
                            [1.0 + v * np.cos(0.05) * dt * 3],
                            [1.0 + v * np.cos(0.05) * dt * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    y=array(
                        [
                            [v * np.sin(0.05) * dt],
                            [v * np.sin(0.05) * dt * 2],
                            [v * np.sin(0.05) * dt * 3],
                            [v * np.sin(0.05) * dt * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    heading=array([[0.05]] * T_p, shape=(T_p, K)),
                ),
            ),
            (  # Bicycle: zero out acceleration, obstacle maintains constant velocity
                create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                    ),
                    estimator=model.bicycle.estimator.finite_difference(
                        time_step_size=dt, wheelbase=L
                    ),
                    prediction=prediction_creator.bicycle(),
                    assumptions=lambda inputs: inputs.zeroed(acceleration=True),
                ),
                data.obstacle_2d_poses(
                    # History shows acceleration: v_0 = 8 m/s, v_1 = 10 m/s, a = 20 m/s²
                    # But with zeroed acceleration, continues at v = 10 m/s
                    x=array([[0.0], [0.8], [x_0 := 1.8]], shape=(T_h := 3, K := 1)),
                    y=array([[0.0], [0.0], [0.0]], shape=(T_h, K)),
                    heading=array([[0.0], [0.0], [0.0]], shape=(T_h, K)),
                ),
                # With zeroed acceleration: constant velocity v = 10 m/s
                data.obstacle_2d_poses(
                    x=array(
                        [
                            [x_0 + (v := 10.0) * dt * 1],
                            [x_0 + v * dt * 2],
                            [x_0 + v * dt * 3],
                            [x_0 + v * dt * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    y=np.full((T_p, K), 0.0),
                    heading=np.full((T_p, K), 0.0),
                ),
            ),
            (  # Bicycle: zero out both acceleration and steering angle
                create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                    ),
                    estimator=model.bicycle.estimator.finite_difference(
                        time_step_size=dt, wheelbase=L
                    ),
                    prediction=prediction_creator.bicycle(),
                    assumptions=lambda inputs: inputs.zeroed(
                        acceleration=True, steering_angle=True
                    ),
                ),
                data.obstacle_2d_poses(
                    # History shows: accelerating and turning
                    # Speed at last step: v = (1.0 * cos(0.05)) / 0.1 ≈ 9.9875 m/s
                    x=array([[0.0], [0.8], [x_0 := 1.8]], shape=(T_h := 3, K := 1)),
                    y=array([[0.0], [0.0], [0.0]], shape=(T_h, K)),
                    heading=array([[0.0], [0.0], [theta := 0.05]], shape=(T_h, K)),
                ),
                # With both zeroed: straight line at constant velocity
                # v = (Δx * cos(θ)) / dt = (1.0 * cos(0.05)) / 0.1
                data.obstacle_2d_poses(
                    x=array(
                        [
                            [
                                x_0
                                + (v := 1.0 * np.cos(theta) / dt) * np.cos(theta) * dt
                            ],
                            [x_0 + v * np.cos(theta) * dt * 2],
                            [x_0 + v * np.cos(theta) * dt * 3],
                            [x_0 + v * np.cos(theta) * dt * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    y=array(
                        [
                            [v * np.sin(theta) * dt * 1],
                            [v * np.sin(theta) * dt * 2],
                            [v * np.sin(theta) * dt * 3],
                            [v * np.sin(theta) * dt * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    heading=np.full((T_p, K), theta),
                ),
            ),
            (  # Unicycle: zero out angular velocity, obstacle goes straight
                create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                    estimator=model.unicycle.estimator.finite_difference(
                        time_step_size=dt
                    ),
                    prediction=prediction_creator.unicycle(),
                    assumptions=lambda inputs: inputs.zeroed(angular_velocity=True),
                ),
                data.obstacle_2d_poses(
                    # Projection-based speed: v = 1.0 * cos(0.05) / 0.1 ≈ 9.9875 m/s
                    # ω = 0.5 rad/s, but will be zeroed by assumption
                    x=array([[0.0], [1.0]], shape=(T_h := 2, K := 1)),
                    y=array([[0.0], [0.0]], shape=(T_h, K)),
                    heading=array([[0.0], [0.05]], shape=(T_h, K)),
                ),
                # With zeroed angular velocity: straight-line at heading=0.05
                # v = Δx * cos(θ) / dt = 1.0 * cos(0.05) / 0.1
                data.obstacle_2d_poses(
                    x=array(
                        [
                            [
                                1.0
                                + (v := 1.0 * np.cos(0.05) / 0.1) * np.cos(0.05) * 0.1
                            ],
                            [1.0 + v * np.cos(0.05) * 0.1 * 2],
                            [1.0 + v * np.cos(0.05) * 0.1 * 3],
                            [1.0 + v * np.cos(0.05) * 0.1 * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    y=array(
                        [
                            [v * np.sin(0.05) * 0.1],
                            [v * np.sin(0.05) * 0.1 * 2],
                            [v * np.sin(0.05) * 0.1 * 3],
                            [v * np.sin(0.05) * 0.1 * 4],
                        ],
                        shape=(T_p, K),
                    ),
                    heading=array([[0.05]] * T_p, shape=(T_p, K)),
                ),
            ),
            (  # Unicycle: zero out linear velocity, obstacle stays in place (spinning)
                create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                    estimator=model.unicycle.estimator.finite_difference(
                        time_step_size=dt
                    ),
                    prediction=prediction_creator.unicycle(),
                    assumptions=lambda inputs: inputs.zeroed(linear_velocity=True),
                ),
                data.obstacle_2d_poses(
                    # v = 10 m/s (will be zeroed), ω = 0.5 rad/s
                    x=array([[0.0], [x_0 := 1.0]], shape=(T_h := 2, K := 1)),
                    y=array([[0.0], [0.0]], shape=(T_h, K)),
                    heading=array([[0.0], [theta_0 := 0.05]], shape=(T_h, K)),
                ),
                # With zeroed linear velocity: stays in place, only heading changes
                data.obstacle_2d_poses(
                    x=np.full((T_p, K), x_0),
                    y=np.full((T_p, K), 0.0),
                    heading=array(
                        [
                            [theta_0 + (w := 0.5) * dt * 1],
                            [theta_0 + w * dt * 2],
                            [theta_0 + w * dt * 3],
                            [theta_0 + w * dt * 4],
                        ],
                        shape=(T_p, K),
                    ),
                ),
            ),
            (  # Unicycle: zero out both linear and angular velocity, obstacle frozen
                create_predictor.curvilinear(
                    horizon=(T_p := 4),
                    model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                    estimator=model.unicycle.estimator.finite_difference(
                        time_step_size=dt
                    ),
                    prediction=prediction_creator.unicycle(),
                    assumptions=lambda inputs: inputs.zeroed(
                        linear_velocity=True, angular_velocity=True
                    ),
                ),
                data.obstacle_2d_poses(
                    # v = 10 m/s, ω = 0.5 rad/s - both will be zeroed
                    x=array([[0.0], [x_0 := 1.0]], shape=(T_h := 2, K := 1)),
                    y=array([[0.0], [y_0 := 0.0]], shape=(T_h, K)),
                    heading=array([[0.0], [theta_0 := 0.05]], shape=(T_h, K)),
                ),
                data.obstacle_2d_poses(
                    x=np.full((T_p, K), x_0),
                    y=np.full((T_p, K), y_0),
                    heading=np.full((T_p, K), theta_0),
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
        assert np.allclose(actual.x(), expected.x(), rtol=1e-2, atol=1e-3)
        assert np.allclose(actual.y(), expected.y(), rtol=1e-2, atol=1e-3)
        assert np.allclose(actual.heading(), expected.heading(), rtol=1e-2, atol=1e-3)


class test_that_covariance_is_more_isotropic_when_turning:
    @staticmethod
    def cases(create_predictor, model, prediction_creator, data) -> Sequence[tuple]:
        return [
            (
                predictor := create_predictor.curvilinear(
                    horizon=(T_p := 10),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                    ),
                    estimator=model.bicycle.estimator.ekf(
                        time_step_size=dt,
                        wheelbase=L,
                        process_noise_covariance=0.01,
                        observation_noise_covariance=0.01,
                    ),
                    prediction=prediction_creator.bicycle(),
                ),
                straight_history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.0]], shape=(2, 1)),
                    heading=array([[0.0], [0.0]], shape=(2, 1)),
                ),
                turning_history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.0]], shape=(2, 1)),
                    heading=array([[0.0], [0.4]], shape=(2, 1)),
                ),
            ),
        ]

    @mark.parametrize(
        ["predictor", "straight_history", "turning_history"],
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
    def test[HistoryT](
        self,
        predictor: ObstacleMotionPredictor[HistoryT, ObstacleStates],
        straight_history: HistoryT,
        turning_history: HistoryT,
    ) -> None:

        straight = np.asarray(predictor.predict(history=straight_history).covariance())
        turning = np.asarray(predictor.predict(history=turning_history).covariance())

        turning_condition = compute.condition_number(turning[-1, :2, :2, 0])
        straight_condition = compute.condition_number(straight[-1, :2, :2, 0])

        assert turning_condition < straight_condition, (
            f"Expected turning covariance to be more isotropic than straight, "
            f"but got condition numbers {turning_condition:.2f} (turning) "
            f"vs {straight_condition:.2f} (straight)."
        )


class test_that_all_backends_produce_matching_predictions:
    @staticmethod
    def cases(model, create_predictor, prediction_creator, data) -> Sequence[tuple]:
        dt = 0.1
        L = 1.0
        T_p = 5

        def bicycle_predictors(*, history_shape: tuple[int, int]):
            T, K = history_shape
            return [
                (
                    create_predictor.numpy.curvilinear(
                        horizon=T_p,
                        model=model.numpy.bicycle.obstacle(
                            time_step_size=dt, wheelbase=L
                        ),
                        estimator=model.numpy.bicycle.estimator.ekf(
                            time_step_size=dt,
                            wheelbase=L,
                            process_noise_covariance=0.01,
                            observation_noise_covariance=0.01,
                        ),
                        prediction=prediction_creator.numpy.bicycle(),
                    ),
                    data.numpy.obstacle_2d_poses(
                        x=array([[0.0, 5.0], [1.0, 6.0]], shape=(T, K)),
                        y=array([[0.0, 0.0], [0.0, 1.0]], shape=(T, K)),
                        heading=array(
                            [[0.0, np.pi / 4], [0.1, np.pi / 4]], shape=(T, K)
                        ),
                    ),
                ),
                (
                    create_predictor.jax.curvilinear(
                        horizon=T_p,
                        model=model.jax.bicycle.obstacle(
                            time_step_size=dt, wheelbase=L
                        ),
                        estimator=model.jax.bicycle.estimator.ekf(
                            time_step_size=dt,
                            wheelbase=L,
                            process_noise_covariance=0.01,
                            observation_noise_covariance=0.01,
                        ),
                        prediction=prediction_creator.jax.bicycle(),
                    ),
                    data.jax.obstacle_2d_poses(
                        x=array([[0.0, 5.0], [1.0, 6.0]], shape=(T, K)),
                        y=array([[0.0, 0.0], [0.0, 1.0]], shape=(T, K)),
                        heading=array(
                            [[0.0, np.pi / 4], [0.1, np.pi / 4]], shape=(T, K)
                        ),
                    ),
                ),
            ]

        def unicycle_predictors(*, history_shape: tuple[int, int]):
            T, K = history_shape
            return [
                (
                    create_predictor.numpy.curvilinear(
                        horizon=T_p,
                        model=model.numpy.unicycle.obstacle(time_step_size=dt),
                        estimator=model.numpy.unicycle.estimator.ekf(
                            time_step_size=dt,
                            process_noise_covariance=0.01,
                            observation_noise_covariance=0.01,
                        ),
                        prediction=prediction_creator.numpy.unicycle(),
                    ),
                    data.numpy.obstacle_2d_poses(
                        x=array([[0.0], [1.0]], shape=(T, K)),
                        y=array([[0.0], [0.5]], shape=(T, K)),
                        heading=array([[0.0], [0.1]], shape=(T, K)),
                    ),
                ),
                (
                    create_predictor.jax.curvilinear(
                        horizon=T_p,
                        model=model.jax.unicycle.obstacle(time_step_size=dt),
                        estimator=model.jax.unicycle.estimator.ekf(
                            time_step_size=dt,
                            process_noise_covariance=0.01,
                            observation_noise_covariance=0.01,
                        ),
                        prediction=prediction_creator.jax.unicycle(),
                    ),
                    data.jax.obstacle_2d_poses(
                        x=array([[0.0], [1.0]], shape=(T, K)),
                        y=array([[0.0], [0.5]], shape=(T, K)),
                        heading=array([[0.0], [0.1]], shape=(T, K)),
                    ),
                ),
            ]

        def integrator_predictors(*, history_shape: tuple[int, int]):
            T, K = history_shape
            return [
                (
                    create_predictor.numpy.curvilinear(
                        horizon=T_p,
                        model=model.numpy.integrator.obstacle(
                            time_step_size=dt, state_dimension=3
                        ),
                        estimator=model.numpy.integrator.estimator.finite_difference(
                            time_step_size=dt
                        ),
                        prediction=prediction_creator.numpy.integrator(),
                    ),
                    data.numpy.obstacle_2d_poses(
                        x=array([[0.0, 1.0], [1.0, 2.0]], shape=(T, K)),
                        y=array([[0.0, 0.0], [1.0, 1.0]], shape=(T, K)),
                        heading=array(
                            [[0.0, np.pi / 6], [0.1, np.pi / 6]], shape=(T, K)
                        ),
                    ),
                ),
                (
                    create_predictor.jax.curvilinear(
                        horizon=T_p,
                        model=model.jax.integrator.obstacle(
                            time_step_size=dt, state_dimension=3
                        ),
                        estimator=model.jax.integrator.estimator.finite_difference(
                            time_step_size=dt
                        ),
                        prediction=prediction_creator.jax.integrator(),
                    ),
                    data.jax.obstacle_2d_poses(
                        x=array([[0.0, 1.0], [1.0, 2.0]], shape=(T, K)),
                        y=array([[0.0, 0.0], [1.0, 1.0]], shape=(T, K)),
                        heading=array(
                            [[0.0, np.pi / 6], [0.1, np.pi / 6]], shape=(T, K)
                        ),
                    ),
                ),
            ]

        return [
            (bicycle_predictors(history_shape=(2, 2)),),
            (unicycle_predictors(history_shape=(2, 1)),),
            (integrator_predictors(history_shape=(2, 2)),),
        ]

    @mark.parametrize(
        ["predictors"],
        cases(
            model=model,
            create_predictor=create_predictor,
            prediction_creator=prediction_creator,
            data=data,
        ),
    )
    def test(
        self, predictors: Sequence[tuple[ObstacleMotionPredictor, object]]
    ) -> None:
        predictions = [
            predictor.predict(history=history) for predictor, history in predictors
        ]

        reference = predictions[0]
        for i, prediction in enumerate(predictions[1:], start=1):
            assert np.allclose(
                np.asarray(reference.x()),
                np.asarray(prediction.x()),
                rtol=1e-4,
                atol=1e-4,
            ), f"X positions do not match between backend 0 and {i}"

            assert np.allclose(
                np.asarray(reference.y()),
                np.asarray(prediction.y()),
                rtol=1e-4,
                atol=1e-4,
            ), f"Y positions do not match between backend 0 and {i}"

            assert np.allclose(
                np.asarray(reference.heading()),
                np.asarray(prediction.heading()),
                rtol=1e-4,
                atol=1e-4,
            ), f"Headings do not match between backend 0 and {i}"

            assert np.allclose(
                np.asarray(reference.covariance()),
                np.asarray(prediction.covariance()),
                rtol=1e-3,
                atol=1e-3,
            ), f"Covariances do not match between backend 0 and {i}"


class test_that_higher_initial_state_covariance_leads_to_higher_prediction_covariance:
    @staticmethod
    def cases(create_predictor, model, prediction_creator, data) -> Sequence[tuple]:
        dt = 0.1
        L = 1.0
        T_p = 5

        return [
            (
                certain_predictor := predictor(covariance_scale=1.0),
                uncertain_predictor := predictor(covariance_scale=20.0),
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(T := 2, K := 1)),
                    y=array([[0.0], [0.0]], shape=(T, K)),
                    heading=array([[0.0], [0.1]], shape=(T, K)),
                ),
            )
            for predictor in [
                lambda covariance_scale, dt=dt, L=L, T_p=T_p: (
                    create_predictor.curvilinear(
                        horizon=T_p,
                        model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                        estimator=model.bicycle.estimator.ekf(
                            time_step_size=dt,
                            wheelbase=L,
                            process_noise_covariance=0.01,
                            observation_noise_covariance=0.01,
                            initial_state_covariance=np.diag(
                                [0.01, 0.01, 0.01, 0.1, 0.1, 0.01]
                            )
                            * covariance_scale,
                        ),
                        prediction=prediction_creator.bicycle(),
                    )
                ),
                lambda covariance_scale, dt=dt, L=L, T_p=T_p: (
                    create_predictor.curvilinear(
                        horizon=T_p,
                        model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                        estimator=model.bicycle.estimator.ukf(
                            time_step_size=dt,
                            wheelbase=L,
                            process_noise_covariance=0.01,
                            observation_noise_covariance=0.01,
                            initial_state_covariance=np.diag(
                                [0.01, 0.01, 0.01, 0.1, 0.1, 0.01]
                            )
                            * covariance_scale,
                        ),
                        prediction=prediction_creator.bicycle(),
                    )
                ),
                lambda covariance_scale, dt=dt, T_p=T_p: create_predictor.curvilinear(
                    horizon=T_p,
                    model=model.unicycle.obstacle(time_step_size=dt),
                    estimator=model.unicycle.estimator.ekf(
                        time_step_size=dt,
                        process_noise_covariance=0.01,
                        observation_noise_covariance=0.01,
                        initial_state_covariance=np.diag([0.01, 0.01, 0.01, 0.1, 0.1])
                        * covariance_scale,
                    ),
                    prediction=prediction_creator.unicycle(),
                ),
                lambda covariance_scale, dt=dt, T_p=T_p: create_predictor.curvilinear(
                    horizon=T_p,
                    model=model.unicycle.obstacle(time_step_size=dt),
                    estimator=model.unicycle.estimator.ukf(
                        time_step_size=dt,
                        process_noise_covariance=0.01,
                        observation_noise_covariance=0.01,
                        initial_state_covariance=np.diag([10.0, 10.0, 1.0, 10.0, 10.0])
                        * covariance_scale,
                    ),
                    prediction=prediction_creator.unicycle(),
                ),
                lambda covariance_scale, dt=dt, T_p=T_p: create_predictor.curvilinear(
                    horizon=T_p,
                    model=model.integrator.obstacle(
                        time_step_size=dt, state_dimension=3
                    ),
                    estimator=model.integrator.estimator.kf(
                        time_step_size=dt,
                        process_noise_covariance=0.01,
                        observation_noise_covariance=0.01,
                        initial_state_covariance=np.diag(
                            [0.01, 0.01, 0.01, 0.1, 0.1, 0.01]
                        )
                        * covariance_scale,
                    ),
                    prediction=prediction_creator.integrator(),
                ),
            ]
        ]

    @mark.parametrize(
        ["certain_predictor", "uncertain_predictor", "history"],
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
    def test[HistoryT](
        self,
        certain_predictor: ObstacleMotionPredictor[HistoryT, ObstacleStates],
        uncertain_predictor: ObstacleMotionPredictor[HistoryT, ObstacleStates],
        history: HistoryT,
    ) -> None:
        certain_prediction = certain_predictor.predict(history=history)
        uncertain_prediction = uncertain_predictor.predict(history=history)

        certain = np.asarray(certain_prediction.covariance())
        uncertain = np.asarray(uncertain_prediction.covariance())

        assert (high := np.trace(uncertain[0, :2, :2, 0])) > (
            low := np.trace(certain[0, :2, :2, 0])
        ), (
            f"Expected higher initial covariance to produce higher prediction "
            f"variance, but got {high:.4f} (high initial cov.) <= {low:.4f} (low initial cov.)."
        )
