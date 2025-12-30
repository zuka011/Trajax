from trajax import (
    model,
    propagator,
    predictor as create_predictor,
    types,
    ObstacleStates,
    ObstacleMotionPredictor,
)

from numtypes import array

import jax.numpy as jnp
import numpy as np

from tests.dsl import mppi as data
from pytest import mark


class NumPyIntegratorPredictionCreator:
    def __call__(
        self,
        *,
        states: types.numpy.integrator.ObstacleStateSequences,
        covariances: types.numpy.PositionCovariance,
    ) -> types.numpy.ObstacleStates:
        return data.numpy.obstacle_states(
            x=states.array[:, 0, :],
            y=states.array[:, 1, :],
            heading=states.array[:, 2, :],
        )

    def empty(self, *, horizon: int) -> types.numpy.ObstacleStates:
        return data.numpy.obstacle_states(
            x=np.empty((horizon, 0)),
            y=np.empty((horizon, 0)),
            heading=np.empty((horizon, 0)),
        )


class NumPyBicyclePredictionCreator:
    def __call__(
        self,
        *,
        states: types.numpy.bicycle.ObstacleStateSequences,
        covariances: types.numpy.PositionCovariance | None,
    ) -> types.numpy.ObstacleStates:
        return data.numpy.obstacle_states(
            x=states.x(),
            y=states.y(),
            heading=states.theta(),
            covariance=covariances,
        )

    def empty(self, *, horizon: int) -> types.numpy.ObstacleStates:
        return data.numpy.obstacle_states(
            x=np.empty((horizon, 0)),
            y=np.empty((horizon, 0)),
            heading=np.empty((horizon, 0)),
        )


class JaxIntegratorPredictionCreator:
    def __call__(
        self,
        *,
        states: types.jax.integrator.ObstacleStateSequences,
        covariances: types.jax.PositionCovariance,
    ) -> types.jax.ObstacleStates:
        return data.jax.obstacle_states(
            x=states.array[:, 0, :],
            y=states.array[:, 1, :],
            heading=states.array[:, 2, :],
        )

    def empty(self, *, horizon: int) -> types.jax.ObstacleStates:
        return data.jax.obstacle_states(
            x=jnp.empty((horizon, 0)),
            y=jnp.empty((horizon, 0)),
            heading=jnp.empty((horizon, 0)),
        )


class JaxBicyclePredictionCreator:
    def __call__(
        self,
        *,
        states: types.jax.bicycle.ObstacleStateSequences,
        covariances: types.jax.PositionCovariance,
    ) -> types.jax.ObstacleStates:
        return data.jax.obstacle_states(
            x=states.x(),
            y=states.y(),
            heading=states.theta(),
            covariance=covariances,
        )

    def empty(self, *, horizon: int) -> types.jax.ObstacleStates:
        return data.jax.obstacle_states(
            x=jnp.empty((horizon, 0)),
            y=jnp.empty((horizon, 0)),
            heading=jnp.empty((horizon, 0)),
        )


@mark.parametrize(
    ["predictor", "history", "expected"],
    [  # Single Integrator CL model tests
        (  # No history
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 5),
                model=model.numpy.integrator.obstacle(time_step_size=(dt := 0.1)),
                prediction=NumPyIntegratorPredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=np.empty((T_h := 0, K := 0)),
                y=np.empty((T_h, K)),
                heading=np.empty((T_h, K)),
            ),
            expected := data.numpy.obstacle_states(
                x=np.empty((T_p, K)),
                y=np.empty((T_p, K)),
                heading=np.empty((T_p, K)),
            ),
        ),
        (  # Single time step history, expected to stay still
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 5),
                model=model.numpy.integrator.obstacle(time_step_size=(dt := 0.1)),
                prediction=NumPyIntegratorPredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=array([[x := -5.0]], shape=(T_h := 1, K := 1)),
                y=array([[y := 2.0]], shape=(T_h, K)),
                heading=array([[theta := 0.0]], shape=(T_h, K)),
            ),
            expected := data.numpy.obstacle_states(
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
                model=model.numpy.integrator.obstacle(time_step_size=(dt := 0.1)),
                prediction=NumPyIntegratorPredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=array(
                    [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], shape=(T_h := 3, K := 2)
                ),
                y=array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], shape=(T_h, K)),
                heading=array(
                    [[0.0, np.pi / 2], [0.0, np.pi / 4], [0.0, 0.0]],
                    shape=(T_h, K),
                ),
            ),
            expected := data.numpy.obstacle_states(
                x=array(
                    [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
                    shape=(T_p, K),  # type: ignore
                ),
                y=array(
                    [[0.0, 4.0], [0.0, 5.0], [0.0, 6.0], [0.0, 7.0]],
                    shape=(T_p, K),  # type: ignore
                ),
                heading=array(
                    [
                        [0.0, -np.pi / 4],
                        [0.0, -np.pi / 2],
                        [0.0, -3 * np.pi / 4],
                        [0.0, -np.pi],
                    ],
                    shape=(T_p, K),  # type: ignore
                ),
            ),
        ),
        (
            # Multiple time steps, stationary obstacle (in last two steps)
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.numpy.integrator.obstacle(time_step_size=(dt := 0.1)),
                prediction=NumPyIntegratorPredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=array(
                    [[-0.5, 0.0], [-1.0, 0.0], [-1.0, 0.0]], shape=(T_h := 3, K := 2)
                ),
                y=array([[0.0, 1.0], [0.0, 2.0], [0.0, 2.0]], shape=(T_h, K)),
                heading=array(
                    [[0.0, np.pi / 2], [0.0, np.pi / 4], [0.0, np.pi / 4]],
                    shape=(T_h, K),
                ),
            ),
            expected := data.numpy.obstacle_states(
                x=array(
                    [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
                    shape=(T_p, K),  # type: ignore
                ),
                y=array(
                    [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]],
                    shape=(T_p, K),  # type: ignore
                ),
                heading=array(
                    [
                        [0.0, np.pi / 4],
                        [0.0, np.pi / 4],
                        [0.0, np.pi / 4],
                        [0.0, np.pi / 4],
                    ],
                    shape=(T_p, K),  # type: ignore
                ),
            ),
        ),
    ]
    + [  # Bicycle CL model tests
        (  # No history
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 5),
                model=model.numpy.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=NumPyBicyclePredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=np.empty((T_h := 0, K := 0)),
                y=np.empty((T_h, K)),
                heading=np.empty((T_h, K)),
            ),
            expected := data.numpy.obstacle_states(
                x=np.empty((T_p, K)),
                y=np.empty((T_p, K)),
                heading=np.empty((T_p, K)),
            ),
        ),
        (  # Single state, zero velocity - stays still
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 5),
                model=model.numpy.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=NumPyBicyclePredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=array([[x := 3.0]], shape=(T_h := 1, K := 1)),
                y=array([[y := 2.0]], shape=(T_h, K)),
                heading=array([[theta := np.pi / 4]], shape=(T_h, K)),
            ),
            expected := data.numpy.obstacle_states(
                x=np.full((T_p, K), x),
                y=np.full((T_p, K), y),
                heading=np.full((T_p, K), theta),
            ),
        ),
        (  # Single state, moving along x-axis (θ=0)
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.numpy.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=NumPyBicyclePredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=array([[-1.0], [4.0]], shape=(T_h := 2, K := 1)),
                y=array([[2.0], [2.0]], shape=(T_h, K)),
                heading=array([[0.0], [0.0]], shape=(T_h, K)),
            ),
            # x increases by 5.0 per step, y stays constant
            expected := data.numpy.obstacle_states(
                x=array([[9.0], [14.0], [19.0], [24.0]], shape=(T_p, K)),
                y=np.full((T_p, K), 2.0),
                heading=np.full((T_p, K), 0.0),
            ),
        ),
        (  # Moving along y-axis (θ=π/2)
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.numpy.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=NumPyBicyclePredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                y=array([[0.0], [5.0]], shape=(T_h, K)),
                heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
            ),
            # y increases by 5.0 per step, x stays constant
            expected := data.numpy.obstacle_states(
                x=np.full((T_p, K), 0.0),
                y=array([[10.0], [15.0], [20.0], [25.0]], shape=(T_p, K)),
                heading=np.full((T_p, K), np.pi / 2),
            ),
        ),
        (  # Multiple time steps, but obstacle is stationary
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.numpy.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=NumPyBicyclePredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                y=array([[1.0], [1.0]], shape=(T_h, K)),
                heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
            ),
            # y increases by 5.0 per step, x stays constant
            expected := data.numpy.obstacle_states(
                x=np.full((T_p, K), 0.0),
                y=np.full((T_p, K), 1.0),
                heading=np.full((T_p, K), np.pi / 2),
            ),
        ),
        (  # Multiple obstacles with different velocities and headings
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 3),
                model=model.numpy.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=NumPyBicyclePredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                # Obstacle 0 - v = 10 m/s, θ=0 (moving +x)
                # Obstacle 1 - v = 10 m/s, θ=π/2 (moving +y)
                # Obstacle 2 - v = 20 m/s, θ=π (moving -x)
                x=array([[0.0, 5.0, 10.0], [1.0, 5.0, 8.0]], shape=(T_h := 2, K := 3)),
                y=array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], shape=(T_h, K)),
                heading=array(
                    [[0.0, np.pi / 2, np.pi], [0.0, np.pi / 2, np.pi]], shape=(T_h, K)
                ),
            ),
            expected := data.numpy.obstacle_states(
                x=array(
                    [[2.0, 5.0, 6.0], [3.0, 5.0, 4.0], [4.0, 5.0, 2.0]], shape=(T_p, K)
                ),
                y=array(
                    [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]], shape=(T_p, K)
                ),
                heading=array([[0.0, np.pi / 2, np.pi]] * T_p, shape=(T_p, K)),
            ),
        ),
        (  # Turning vehicle - constant steering angle (δ) preserved
            # θ̇ = (v/L) tan(δ), so constant δ means constant angular velocity ω
            # From history: estimate v and ω, then δ = arctan(ω * L / v)
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.numpy.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                ),
                prediction=NumPyBicyclePredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
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
            expected := data.numpy.obstacle_states(
                x=array(
                    [
                        [1.0 + 10 * np.cos(0.05) * 0.1],
                        [1.0 + 10 * np.cos(0.05) * 0.1 + 10 * np.cos(0.10) * 0.1],
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
    ]
    + [  # Analogous tests for JAX Integrator CL model
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 5),
                model=model.jax.integrator.obstacle(time_step_size=(dt := 0.1)),
                prediction=JaxIntegratorPredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=np.empty((T_h := 0, K := 0)),
                y=np.empty((T_h, K)),
                heading=np.empty((T_h, K)),
            ),
            expected := data.jax.obstacle_states(
                x=np.empty((T_p, K)),
                y=np.empty((T_p, K)),
                heading=np.empty((T_p, K)),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 5),
                model=model.jax.integrator.obstacle(time_step_size=(dt := 0.1)),
                prediction=JaxIntegratorPredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array([[x := -5.0]], shape=(T_h := 1, K := 1)),
                y=array([[y := 2.0]], shape=(T_h, K)),
                heading=array([[theta := 0.0]], shape=(T_h, K)),
            ),
            expected := data.jax.obstacle_states(
                x=np.full((T_p, K), x),
                y=np.full((T_p, K), y),
                heading=np.full((T_p, K), theta),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.jax.integrator.obstacle(time_step_size=(dt := 0.1)),
                prediction=JaxIntegratorPredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array(
                    [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], shape=(T_h := 3, K := 2)
                ),
                y=array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], shape=(T_h, K)),
                heading=array(
                    [[0.0, np.pi / 2], [0.0, np.pi / 4], [0.0, 0.0]],
                    shape=(T_h, K),
                ),
            ),
            expected := data.jax.obstacle_states(
                x=array(
                    [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
                    shape=(T_p, K),  # type: ignore
                ),
                y=array(
                    [[0.0, 4.0], [0.0, 5.0], [0.0, 6.0], [0.0, 7.0]],
                    shape=(T_p, K),  # type: ignore
                ),
                heading=array(
                    [
                        [0.0, -np.pi / 4],
                        [0.0, -np.pi / 2],
                        [0.0, -3 * np.pi / 4],
                        [0.0, -np.pi],
                    ],
                    shape=(T_p, K),  # type: ignore
                ),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.jax.integrator.obstacle(time_step_size=(dt := 0.1)),
                prediction=JaxIntegratorPredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array(
                    [[-0.5, 0.0], [-1.0, 0.0], [-1.0, 0.0]], shape=(T_h := 3, K := 2)
                ),
                y=array([[0.0, 1.0], [0.0, 2.0], [0.0, 2.0]], shape=(T_h, K)),
                heading=array(
                    [[0.0, np.pi / 2], [0.0, np.pi / 4], [0.0, np.pi / 4]],
                    shape=(T_h, K),
                ),
            ),
            expected := data.jax.obstacle_states(
                x=array(
                    [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
                    shape=(T_p, K),  # type: ignore
                ),
                y=array(
                    [[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]],
                    shape=(T_p, K),  # type: ignore
                ),
                heading=array(
                    [
                        [0.0, np.pi / 4],
                        [0.0, np.pi / 4],
                        [0.0, np.pi / 4],
                        [0.0, np.pi / 4],
                    ],
                    shape=(T_p, K),  # type: ignore
                ),
            ),
        ),
    ]
    + [  # JAX Bicycle CL model tests
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 5),
                model=model.jax.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=JaxBicyclePredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=np.empty((T_h := 0, K := 0)),
                y=np.empty((T_h, K)),
                heading=np.empty((T_h, K)),
            ),
            expected := data.jax.obstacle_states(
                x=np.empty((T_p, K)),
                y=np.empty((T_p, K)),
                heading=np.empty((T_p, K)),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 5),
                model=model.jax.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=JaxBicyclePredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array([[x := 3.0]], shape=(T_h := 1, K := 1)),
                y=array([[y := 2.0]], shape=(T_h, K)),
                heading=array([[theta := np.pi / 4]], shape=(T_h, K)),
            ),
            expected := data.jax.obstacle_states(
                x=np.full((T_p, K), x),
                y=np.full((T_p, K), y),
                heading=np.full((T_p, K), theta),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.jax.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=JaxBicyclePredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array([[-1.0], [4.0]], shape=(T_h := 2, K := 1)),
                y=array([[2.0], [2.0]], shape=(T_h, K)),
                heading=array([[0.0], [0.0]], shape=(T_h, K)),
            ),
            # x increases by 5.0 per step, y stays constant
            expected := data.jax.obstacle_states(
                x=array([[9.0], [14.0], [19.0], [24.0]], shape=(T_p, K)),
                y=np.full((T_p, K), 2.0),
                heading=np.full((T_p, K), 0.0),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.jax.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=JaxBicyclePredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                y=array([[0.0], [5.0]], shape=(T_h, K)),
                heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
            ),
            expected := data.jax.obstacle_states(
                x=np.full((T_p, K), 0.0),
                y=array([[10.0], [15.0], [20.0], [25.0]], shape=(T_p, K)),
                heading=np.full((T_p, K), np.pi / 2),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.jax.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=JaxBicyclePredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                y=array([[1.0], [1.0]], shape=(T_h, K)),
                heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
            ),
            expected := data.jax.obstacle_states(
                x=np.full((T_p, K), 0.0),
                y=np.full((T_p, K), 1.0),
                heading=np.full((T_p, K), np.pi / 2),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 3),
                model=model.jax.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=JaxBicyclePredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array([[0.0, 5.0, 10.0], [1.0, 5.0, 8.0]], shape=(T_h := 2, K := 3)),
                y=array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], shape=(T_h, K)),
                heading=array(
                    [[0.0, np.pi / 2, np.pi], [0.0, np.pi / 2, np.pi]], shape=(T_h, K)
                ),
            ),
            expected := data.jax.obstacle_states(
                x=array(
                    [[2.0, 5.0, 6.0], [3.0, 5.0, 4.0], [4.0, 5.0, 2.0]], shape=(T_p, K)
                ),
                y=array(
                    [[0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 4.0, 0.0]], shape=(T_p, K)
                ),
                heading=array([[0.0, np.pi / 2, np.pi]] * T_p, shape=(T_p, K)),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.jax.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                ),
                prediction=JaxBicyclePredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array([[0.0], [1.0]], shape=(T_h := 2, K := 1)),
                y=array([[0.0], [0.0]], shape=(T_h, K)),
                heading=array([[0.0], [0.05]], shape=(T_h, K)),
            ),
            expected := data.jax.obstacle_states(
                x=array(
                    [
                        [1.0 + 10 * np.cos(0.05) * 0.1],
                        [1.0 + 10 * np.cos(0.05) * 0.1 + 10 * np.cos(0.10) * 0.1],
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
)
def test_that_obstacle_motion_is_predicted_correctly[
    HistoryT,
    PredictionT: ObstacleStates,
](
    predictor: ObstacleMotionPredictor[HistoryT, PredictionT],
    history: HistoryT,
    expected: PredictionT,
) -> None:
    actual = predictor.predict(history=history)
    assert np.allclose(actual.x(), expected.x(), rtol=1e-3, atol=1e-6)
    assert np.allclose(actual.y(), expected.y(), rtol=1e-3, atol=1e-6)
    assert np.allclose(actual.heading(), expected.heading(), rtol=1e-3, atol=1e-6)


@mark.parametrize(
    ["predictor", "history"],
    [
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.numpy.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=NumPyBicyclePredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                y=array([[1.0], [1.0]], shape=(T_h, K)),
                heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.jax.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                prediction=JaxBicyclePredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                y=array([[1.0], [1.0]], shape=(T_h, K)),
                heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
            ),
        ),
    ],
)
def test_that_no_covariance_information_is_provided_when_propagator_is_not_available[
    HistoryT,
    PredictionT: ObstacleStates,
](predictor: ObstacleMotionPredictor[HistoryT, PredictionT], history: HistoryT) -> None:
    assert predictor.predict(history=history).covariance() is None


@mark.parametrize(
    ["predictor", "history"],
    [
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.numpy.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                propagator=propagator.numpy.linear(
                    time_step_size=dt,
                    initial_covariance=propagator.numpy.covariance.constant_variance(
                        position_variance=0.1, velocity_variance=0.2
                    ),
                    padding=propagator.padding(to_dimension=3, epsilon=1e-15),
                ),
                prediction=NumPyBicyclePredictionCreator(),
            ),
            history := data.numpy.obstacle_states(
                x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                y=array([[1.0], [1.0]], shape=(T_h, K)),
                heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
            ),
        ),
        (
            predictor := create_predictor.curvilinear(
                horizon=(T_p := 4),
                model=model.jax.bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                propagator=propagator.jax.linear(
                    time_step_size=dt,
                    initial_covariance=propagator.jax.covariance.constant_variance(
                        position_variance=0.1, velocity_variance=0.2
                    ),
                    padding=propagator.padding(to_dimension=3, epsilon=1e-15),
                ),
                prediction=JaxBicyclePredictionCreator(),
            ),
            history := data.jax.obstacle_states(
                x=array([[0.0], [0.0]], shape=(T_h := 2, K := 1)),
                y=array([[1.0], [1.0]], shape=(T_h, K)),
                heading=array([[np.pi / 2], [np.pi / 2]], shape=(T_h, K)),
            ),
        ),
    ],
)
def test_that_position_covariance_information_is_provided_when_propagator_is_available[
    HistoryT,
    PredictionT: ObstacleStates,
](predictor: ObstacleMotionPredictor[HistoryT, PredictionT], history: HistoryT) -> None:
    covariances = np.asarray(predictor.predict(history=history).covariance())

    assert np.all(
        [
            (covariances[t + 1, 0, 0] > covariances[t, 0, 0])
            & (covariances[t + 1, 1, 1] > covariances[t, 1, 1])
            for t in range(T_p - 1)
        ]
    )
