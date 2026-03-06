from typing import Sequence

from faran import (
    ObstacleStates,
    ObstacleMotionPredictor,
    predictor as create_predictor,
)

from numtypes import array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


class test_that_state_of_obstacle_is_predicted_to_remain_constant:
    @staticmethod
    def cases(create_predictor, data) -> Sequence[tuple]:
        return [
            (  # Single observation
                predictor := create_predictor.static(horizon=(T_p := 5)),
                history := data.obstacle_2d_poses(
                    x=array([[x := -5.0]], shape=(T_h := 1, K := 1)),
                    y=array([[y := 2.0]], shape=(T_h, K)),
                    heading=array([[theta := 0.3]], shape=(T_h, K)),
                ),
                expected := data.obstacle_2d_poses(
                    x=np.full((T_p, K), x),
                    y=np.full((T_p, K), y),
                    heading=np.full((T_p, K), theta),
                ),
            ),
            (  # Multiple obstacles — last observation is expected to be the steady state
                predictor := create_predictor.static(horizon=(T_p := 3)),
                history := data.obstacle_2d_poses(
                    x=array(
                        [[0.0, 5.0, -2.0], [1.0, 4.0, -3.0]],
                        shape=(T_h := 2, K := 3),
                    ),
                    y=array(
                        [[1.0, 0.0, 3.0], [2.0, -1.0, 4.0]],
                        shape=(T_h, K),
                    ),
                    heading=array(
                        [[0.0, np.pi / 2, np.pi], [0.1, np.pi / 4, np.pi]],
                        shape=(T_h, K),
                    ),
                ),
                expected := data.obstacle_2d_poses(
                    x=array([[1.0, 4.0, -3.0]] * T_p, shape=(T_p, K)),
                    y=array([[2.0, -1.0, 4.0]] * T_p, shape=(T_p, K)),
                    heading=array([[0.1, np.pi / 4, np.pi]] * T_p, shape=(T_p, K)),
                ),
            ),
        ]

    @mark.parametrize(
        ["predictor", "history", "expected"],
        [
            *cases(create_predictor=create_predictor.numpy, data=data.numpy),
            *cases(create_predictor=create_predictor.jax, data=data.jax),
        ],
    )
    def test[HistoryT, PredictionT: ObstacleStates](
        self,
        predictor: ObstacleMotionPredictor[HistoryT, PredictionT],
        history: HistoryT,
        expected: PredictionT,
    ) -> None:
        actual = predictor.predict(history=history)

        assert np.allclose(actual.x(), expected.x(), atol=1e-10)
        assert np.allclose(actual.y(), expected.y(), atol=1e-10)
        assert np.allclose(actual.heading(), expected.heading(), atol=1e-10)
