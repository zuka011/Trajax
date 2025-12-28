from trajax import (
    model,
    predictor as create_predictor,
    types,
    ObstacleStates,
    ObstacleMotionPredictor,
)

from numtypes import array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


def numpy_empty_prediction(*, horizon: int) -> types.numpy.ObstacleStates:
    return data.numpy.obstacle_states(
        x=np.empty((horizon, 0)),
        y=np.empty((horizon, 0)),
        heading=np.empty((horizon, 0)),
    )


@mark.parametrize(
    ["predictor", "states", "expected"],
    [  # Single Integrator CVM tests
        (  # No history
            predictor := create_predictor.constant_velocity(
                horizon=(T_p := 5),
                model=model.numpy.integrator.obstacle(time_step_size=(dt := 0.1)),
                empty_prediction=numpy_empty_prediction,
            ),
            states := data.numpy.obstacle_states(
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
            predictor := create_predictor.constant_velocity(
                horizon=(T_p := 5),
                model=model.numpy.integrator.obstacle(time_step_size=(dt := 0.1)),
                empty_prediction=numpy_empty_prediction,
            ),
            states := data.numpy.obstacle_states(
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
            predictor := create_predictor.constant_velocity(
                horizon=(T_p := 4),
                model=model.numpy.integrator.obstacle(time_step_size=(dt := 0.1)),
                empty_prediction=numpy_empty_prediction,
            ),
            states := data.numpy.obstacle_states(
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
    ]
    + [  # Bicycle Model CVM tests
        (  # No history
            predictor := create_predictor.constant_velocity(
                horizon=(T_p := 5),
                model=model.numpy.kinematic_bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                empty_prediction=numpy_empty_prediction,
            ),
            states := data.numpy.obstacle_states(
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
            predictor := create_predictor.constant_velocity(
                horizon=(T_p := 5),
                model=model.numpy.kinematic_bicycle.obstacle(
                    time_step_size=(dt := 0.1), wheelbase=1.0
                ),
                empty_prediction=numpy_empty_prediction,
            ),
            states := data.numpy.obstacle_states(
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
    ],
)
def test_that_obstacle_motion_is_predicted_correctly[
    HistoryT,
    PredictionT: ObstacleStates,
](
    predictor: ObstacleMotionPredictor[HistoryT, PredictionT],
    states: HistoryT,
    expected: PredictionT,
) -> None:
    actual = predictor.predict(history=states)
    assert np.allclose(actual.x(), expected.x())
    assert np.allclose(actual.y(), expected.y())
    assert np.allclose(actual.heading(), expected.heading())
