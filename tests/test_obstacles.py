from typing import Any, Sequence

from trajax import obstacles, PredictingObstacleStateProvider

import numpy as np

from tests.dsl import stubs, mppi as data
from pytest import mark


@mark.parametrize(
    ["provider", "states_sequence", "expected"],
    [
        (
            provider := obstacles.numpy.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.numpy.obstacle_states(
                        x=np.random.rand(T := 15, K := 5),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.numpy.obstacle_states(
                            x=np.random.rand(H := 4, K),
                            y=np.random.rand(H, K),
                            heading=np.random.rand(H, K),
                        )
                    ),
                ),
            ),
            states_sequence := [
                history.at(time_step=0),
                history.at(time_step=1),
                history.at(time_step=2),
                history.at(time_step=3),
            ],
            prediction,
        ),
        (
            provider := obstacles.jax.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.jax.obstacle_states(
                        x=np.random.rand(T := 15, K := 5),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.jax.obstacle_states(
                            x=np.random.rand(H := 4, K),
                            y=np.random.rand(H, K),
                            heading=np.random.rand(H, K),
                        )
                    ),
                ),
            ),
            states_sequence := [
                history.at(time_step=0),
                history.at(time_step=1),
                history.at(time_step=2),
                history.at(time_step=3),
            ],
            prediction,
        ),
    ],
)
def test_that_obstacle_state_provider_provides_forecasts_from_obstacle_motion_predictor[
    ObstacleStatesForTimeStepT,
    PredictionT,
](
    provider: PredictingObstacleStateProvider[
        ObstacleStatesForTimeStepT, Any, PredictionT
    ],
    states_sequence: Sequence[ObstacleStatesForTimeStepT],
    expected: PredictionT,
) -> None:
    for states in states_sequence:
        provider.observe(states)

    assert np.allclose(provider(), expected)
