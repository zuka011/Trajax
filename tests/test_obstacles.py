from typing import Any, Sequence
from functools import partial

from trajax import (
    PredictingObstacleStateProvider,
    ObstacleStates,
    ObstacleStatesRunningHistory,
    ObstacleIds,
    types,
    obstacles,
)

from numtypes import array

import numpy as np

from tests.dsl import stubs, mppi as data
from pytest import mark


class test_that_obstacle_state_provider_provides_forecasts_from_obstacle_motion_predictor:
    @staticmethod
    def cases(obstacles, data, types) -> Sequence[tuple]:
        return [
            (
                provider := obstacles.provider.predicting(
                    predictor=stubs.ObstacleMotionPredictor.returns(
                        prediction := data.obstacle_2d_poses(
                            x=np.random.rand(T := 15, K := 5),
                            y=np.random.rand(T, K),
                            heading=np.random.rand(T, K),
                            covariance=np.random.rand(T, 3, 3, K),
                        ),
                        when_history_is=(
                            history := data.obstacle_2d_poses(
                                x=np.random.rand(H := 4, K),
                                y=np.random.rand(H, K),
                                heading=np.random.rand(H, K),
                            )
                        ),
                    ),
                    history=types.obstacle_states_running_history.empty(
                        creator=types.obstacle_2d_poses
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
            *[
                (  # No History
                    provider := obstacles.provider.predicting(
                        predictor=stubs.ObstacleMotionPredictor.returns(
                            prediction := data.obstacle_2d_poses(
                                x=np.random.rand(T := 10, K := 3),
                                y=np.random.rand(T, K),
                                heading=np.random.rand(T, K),
                                covariance=np.random.rand(T, 3, 3, K),
                            ),
                            when_history_is=(
                                history := data.obstacle_2d_poses(
                                    x=np.full(output_shape, np.nan),
                                    y=np.full(output_shape, np.nan),
                                    heading=np.full(output_shape, np.nan),
                                )
                            ),
                        ),
                        history=types.obstacle_states_running_history.empty(
                            creator=types.obstacle_2d_poses,
                            horizon=horizon,
                            obstacle_count=obstacle_count,
                        ),
                    ),
                    states_sequence := [],
                    prediction,
                )
                for horizon, obstacle_count, output_shape in [
                    (None, None, (0, 0)),  # No fixed horizon or obstacle count
                    (4, None, (4, 0)),  # Just horizon fixed
                    (None, 5, (0, 5)),  # Just obstacle count fixed
                    (6, 3, (6, 3)),  # Both horizon and obstacle count fixed
                ]
            ],
            (  # Old observations should be dropped when horizon is exceeded
                provider := obstacles.provider.predicting(
                    predictor=stubs.ObstacleMotionPredictor.returns(
                        prediction := data.obstacle_2d_poses(
                            x=np.random.rand(T := 15, K := 5),
                            y=np.random.rand(T, K),
                            heading=np.random.rand(T, K),
                            covariance=np.random.rand(T, 3, 3, K),
                        ),
                        when_history_is=(
                            history := data.obstacle_2d_poses(
                                x=np.random.rand(H := 4, K),
                                y=np.random.rand(H, K),
                                heading=np.random.rand(H, K),
                            )
                        ),
                    ),
                    history=types.obstacle_states_running_history.empty(
                        creator=types.obstacle_2d_poses, horizon=H
                    ),
                ),
                states_sequence := [
                    *[
                        data.obstacle_2d_poses_for_time_step(
                            x=np.random.rand(K := 5),
                            y=np.random.rand(K),
                            heading=np.random.rand(K),
                        )
                        for _ in range(10)  # This will be discarded
                    ],
                    history.at(time_step=0),
                    history.at(time_step=1),
                    history.at(time_step=2),
                    history.at(time_step=3),
                ],
                prediction,
            ),
        ]

    @mark.parametrize(
        ["provider", "states_sequence", "expected"],
        [
            *cases(obstacles=obstacles.numpy, data=data.numpy, types=types.numpy),
            *cases(obstacles=obstacles.jax, data=data.jax, types=types.jax),
        ],
    )
    def test[ObstacleStatesForTimeStepT, PredictionT: ObstacleStates](
        self,
        provider: PredictingObstacleStateProvider[
            ObstacleStatesForTimeStepT, Any, Any, PredictionT
        ],
        states_sequence: Sequence[ObstacleStatesForTimeStepT],
        expected: PredictionT,
    ) -> None:
        for states in states_sequence:
            provider.observe(states)

        assert np.allclose(provider(), expected)


class test_that_obstacle_state_provider_uses_specified_id_assignment:
    @staticmethod
    def cases(obstacles, data, types) -> Sequence[tuple]:
        return [
            (
                provider := obstacles.provider.predicting(
                    predictor=stubs.ObstacleMotionPredictor.returns(
                        prediction := data.obstacle_2d_poses(
                            x=np.random.rand(T := 15, K := 3),
                            y=np.random.rand(T, K),
                            heading=np.random.rand(T, K),
                            covariance=np.random.rand(T, 3, 3, K),
                        ),
                        when_history_is=(
                            history := data.obstacle_2d_poses(
                                x=(
                                    x := array(
                                        [
                                            x_0 := [
                                                x_02 := 0.1,
                                                x_03 := 0.3,
                                                x_06 := 0.5,
                                            ],
                                            x_1 := [
                                                x_12 := 0.9,
                                                x_13 := 0.2,
                                                x_16 := 0.7,
                                            ],
                                            x_2 := [
                                                x_22 := 0.8,
                                                x_23 := 0.4,
                                                x_26 := 0.6,
                                            ],
                                            x_3 := [
                                                x_32 := 0.6,
                                                x_33 := 0.2,
                                                x_36 := 0.1,
                                            ],
                                        ],
                                        shape=(H := 4, K := 3),
                                    )
                                ),
                                y=(
                                    y := array(
                                        [
                                            y_0 := [
                                                y_02 := 1.2,
                                                y_03 := 4.3,
                                                y_06 := 1.2,
                                            ],
                                            y_1 := [
                                                y_12 := 1.5,
                                                y_13 := 0.8,
                                                y_16 := 1.9,
                                            ],
                                            y_2 := [
                                                y_22 := 1.3,
                                                y_23 := 2.4,
                                                y_26 := 1.4,
                                            ],
                                            y_3 := [
                                                y_32 := 1.7,
                                                y_33 := 1.2,
                                                y_36 := 1.6,
                                            ],
                                        ],
                                        shape=(H, K),
                                    )
                                ),
                                heading=(
                                    heading := array(
                                        [
                                            h_0 := [
                                                h_02 := 2.1,
                                                h_03 := 0.8,
                                                h_06 := 0.3,
                                            ],
                                            h_1 := [
                                                h_12 := 0.4,
                                                h_13 := 4.2,
                                                h_16 := 4.3,
                                            ],
                                            h_2 := [
                                                h_22 := 1.2,
                                                h_23 := 0.3,
                                                h_26 := 0.5,
                                            ],
                                            h_3 := [
                                                h_32 := 5.2,
                                                h_33 := 0.4,
                                                h_36 := 1.4,
                                            ],
                                        ],
                                        shape=(H, K),
                                    )
                                ),
                            )
                        ),
                    ),
                    history=types.obstacle_states_running_history.empty(
                        creator=types.obstacle_2d_poses
                    ),
                    id_assignment=stubs.ObstacleIdAssignment.returns(
                        partial(
                            lambda observation_0, observation_1, observation_2, observation_3, x, y, heading, ids: (
                                # IDs don't start from 0 intentionally.
                                ids(
                                    data.obstacle_ids([6, 2, 3]),
                                    when_observing=observation_0,
                                    and_history=data.obstacle_2d_poses(
                                        x=np.empty((0, 0)),
                                        y=np.empty((0, 0)),
                                        heading=np.empty((0, 0)),
                                    ),
                                    and_ids=data.obstacle_ids([]),
                                ),
                                ids(
                                    data.obstacle_ids([3, 6, 2]),
                                    when_observing=observation_1,
                                    and_history=data.obstacle_2d_poses(
                                        x=x[:1], y=y[:1], heading=heading[:1]
                                    ),
                                    and_ids=data.obstacle_ids([2, 3, 6]),
                                ),
                                ids(
                                    data.obstacle_ids([2, 3, 6]),
                                    when_observing=observation_2,
                                    and_history=data.obstacle_2d_poses(
                                        x=x[:2], y=y[:2], heading=heading[:2]
                                    ),
                                    and_ids=data.obstacle_ids([2, 3, 6]),
                                ),
                                ids(
                                    data.obstacle_ids([3, 2, 6]),
                                    when_observing=observation_3,
                                    and_history=data.obstacle_2d_poses(
                                        x=x[:3], y=y[:3], heading=heading[:3]
                                    ),
                                    and_ids=data.obstacle_ids([2, 3, 6]),
                                ),
                            ),
                            observation_0 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_06, x_02, x_03], shape=(K,)),
                                y=array([y_06, y_02, y_03], shape=(K,)),
                                heading=array([h_06, h_02, h_03], shape=(K,)),
                            ),
                            observation_1 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_13, x_16, x_12], shape=(K,)),
                                y=array([y_13, y_16, y_12], shape=(K,)),
                                heading=array([h_13, h_16, h_12], shape=(K,)),
                            ),
                            observation_2 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_22, x_23, x_26], shape=(K,)),
                                y=array([y_22, y_23, y_26], shape=(K,)),
                                heading=array([h_22, h_23, h_26], shape=(K,)),
                            ),
                            observation_3 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_33, x_32, x_36], shape=(K,)),
                                y=array([y_33, y_32, y_36], shape=(K,)),
                                heading=array([h_33, h_32, h_36], shape=(K,)),
                            ),
                            x,
                            y,
                            heading,
                        )
                    ),
                ),
                states_sequence := [
                    observation_0,
                    observation_1,
                    observation_2,
                    observation_3,
                ],
                prediction,
            ),
            (  # Obstacle disappears then reappears
                provider := obstacles.provider.predicting(
                    predictor=stubs.ObstacleMotionPredictor.returns(
                        prediction := data.obstacle_2d_poses(
                            x=np.random.rand(T := 15, K := 4),
                            y=np.random.rand(T, K),
                            heading=np.random.rand(T, K),
                            covariance=np.random.rand(T, 3, 3, K),
                        ),
                        when_history_is=(
                            history := data.obstacle_2d_poses(
                                x=array(
                                    [
                                        # IDs: (2, 3, 5, 6).
                                        # ID = 3 Disappears
                                        # ID = 5 Appears later
                                        # ID = 6 Disappears then reappears
                                        [x_02 := 0.4, x_03 := 0.3, np.nan, x_06 := 0.6],
                                        [x_12 := 0.2, np.nan, np.nan, np.nan],
                                        [x_22 := 0.3, np.nan, x_25 := 0.6, np.nan],
                                        [x_32 := 0.7, np.nan, x_35 := 0.4, x_36 := 0.2],
                                    ],
                                    shape=(H := 4, K := 4),
                                ),
                                y=array(
                                    [
                                        [y_02 := 1.2, y_03 := 1.3, np.nan, y_06 := 1.6],
                                        [y_12 := 1.1, np.nan, np.nan, np.nan],
                                        [y_22 := 1.3, np.nan, y_25 := 1.6, np.nan],
                                        [y_32 := 1.5, np.nan, y_35 := 1.4, y_36 := 1.2],
                                    ],
                                    shape=(H, K),
                                ),
                                heading=array(
                                    [
                                        [h_02 := 0.1, h_03 := 0.2, np.nan, h_06 := 0.4],
                                        [h_12 := 0.6, np.nan, np.nan, np.nan],
                                        [h_22 := 0.2, np.nan, h_25 := 0.4, np.nan],
                                        [h_32 := 0.3, np.nan, h_35 := 0.3, h_36 := 0.1],
                                    ],
                                    shape=(H, K),
                                ),
                            )
                        ),
                    ),
                    history=types.obstacle_states_running_history.empty(
                        creator=types.obstacle_2d_poses, obstacle_count=K
                    ),
                    id_assignment=stubs.ObstacleIdAssignment.returns(
                        partial(
                            lambda observation_0, history_0, ids_0, observation_1, history_1, ids_1, observation_2, history_2, ids_2, observation_3, history_3, ids_3, ids: (
                                ids(
                                    data.obstacle_ids([6, 3, 2]),
                                    when_observing=observation_0,
                                    and_history=history_0,
                                    and_ids=ids_0,
                                ),
                                ids(
                                    data.obstacle_ids([2]),
                                    when_observing=observation_1,
                                    and_history=history_1,
                                    and_ids=ids_1,
                                ),
                                ids(
                                    data.obstacle_ids([5, 2]),
                                    when_observing=observation_2,
                                    and_history=history_2,
                                    and_ids=ids_2,
                                ),
                                ids(
                                    data.obstacle_ids([2, 5, 6]),
                                    when_observing=observation_3,
                                    and_history=history_3,
                                    and_ids=ids_3,
                                ),
                            ),
                            observation_0 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_06, x_03, x_02], shape=(K_0 := 3,)),
                                y=array([y_06, y_03, y_02], shape=(K_0,)),
                                heading=array([h_06, h_03, h_02], shape=(K_0,)),
                            ),
                            history_0 := data.obstacle_2d_poses(
                                x=np.empty((0, K)),
                                y=np.empty((0, K)),
                                heading=np.empty((0, K)),
                            ),
                            ids_0 := data.obstacle_ids([]),
                            observation_1 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_12], shape=(K_1 := 1,)),
                                y=array([y_12], shape=(K_1,)),
                                heading=array([h_12], shape=(K_1,)),
                            ),
                            # Obstacle indices always match ID indices.
                            history_1 := data.obstacle_2d_poses(
                                x=array([[x_02, x_03, x_06, np.nan]], shape=(1, K)),
                                y=array([[y_02, y_03, y_06, np.nan]], shape=(1, K)),
                                heading=array(
                                    [[h_02, h_03, h_06, np.nan]], shape=(1, K)
                                ),
                            ),
                            ids_1 := data.obstacle_ids([2, 3, 6]),
                            observation_2 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_25, x_22], shape=(K_2 := 2,)),
                                y=array([y_25, y_22], shape=(K_2,)),
                                heading=array([h_25, h_22], shape=(K_2,)),
                            ),
                            history_2 := data.obstacle_2d_poses(
                                x=array(
                                    [
                                        [x_02, x_03, x_06, np.nan],
                                        [x_12, np.nan, np.nan, np.nan],
                                    ],
                                    shape=(2, K),
                                ),
                                y=array(
                                    [
                                        [y_02, y_03, y_06, np.nan],
                                        [y_12, np.nan, np.nan, np.nan],
                                    ],
                                    shape=(2, K),
                                ),
                                heading=array(
                                    [
                                        [h_02, h_03, h_06, np.nan],
                                        [h_12, np.nan, np.nan, np.nan],
                                    ],
                                    shape=(2, K),
                                ),
                            ),
                            ids_2 := data.obstacle_ids([2, 3, 6]),
                            observation_3 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_32, x_35, x_36], shape=(K_3 := 3,)),
                                y=array([y_32, y_35, y_36], shape=(K_3,)),
                                heading=array([h_32, h_35, h_36], shape=(K_3,)),
                            ),
                            # Some historical data will be shifted to match ID indices.
                            history_3 := data.obstacle_2d_poses(
                                x=array(
                                    [
                                        [x_02, x_03, np.nan, x_06],
                                        [x_12, np.nan, np.nan, np.nan],
                                        [x_22, np.nan, x_25, np.nan],
                                    ],
                                    shape=(3, K),
                                ),
                                y=array(
                                    [
                                        [y_02, y_03, np.nan, y_06],
                                        [y_12, np.nan, np.nan, np.nan],
                                        [y_22, np.nan, y_25, np.nan],
                                    ],
                                    shape=(3, K),
                                ),
                                heading=array(
                                    [
                                        [h_02, h_03, np.nan, h_06],
                                        [h_12, np.nan, np.nan, np.nan],
                                        [h_22, np.nan, h_25, np.nan],
                                    ],
                                    shape=(3, K),
                                ),
                            ),
                            ids_3 := data.obstacle_ids([2, 3, 5, 6]),
                        )
                    ),
                ),
                states_sequence := [
                    observation_0,
                    observation_1,
                    observation_2,
                    observation_3,
                ],
                prediction,
            ),
            (  # Total number of IDs is less than expected.
                provider := obstacles.provider.predicting(
                    predictor=stubs.ObstacleMotionPredictor.returns(
                        prediction := data.obstacle_2d_poses(
                            x=np.random.rand(T := 15, K := 4),
                            y=np.random.rand(T, K),
                            heading=np.random.rand(T, K),
                            covariance=np.random.rand(T, 3, 3, K),
                        ),
                        when_history_is=(
                            history := data.obstacle_2d_poses(
                                x=array(
                                    [
                                        # IDs: (2, 3).
                                        [x_02 := 0.2, x_03 := 0.3, np.nan, np.nan],
                                        [x_12 := 0.1, np.nan, np.nan, np.nan],
                                        [np.nan, x_23 := 0.4, np.nan, np.nan],
                                    ],
                                    shape=(H := 3, K := 4),
                                ),
                                y=array(
                                    [
                                        [y_02 := 1.2, y_03 := 1.3, np.nan, np.nan],
                                        [y_12 := 1.1, np.nan, np.nan, np.nan],
                                        [np.nan, y_23 := 1.4, np.nan, np.nan],
                                    ],
                                    shape=(H, K),
                                ),
                                heading=array(
                                    [
                                        [h_02 := 0.1, h_03 := 0.2, np.nan, np.nan],
                                        [h_12 := 0.4, np.nan, np.nan, np.nan],
                                        [np.nan, h_23 := 0.3, np.nan, np.nan],
                                    ],
                                    shape=(H, K),
                                ),
                            )
                        ),
                    ),
                    history=types.obstacle_states_running_history.empty(
                        creator=types.obstacle_2d_poses, obstacle_count=K
                    ),
                    id_assignment=stubs.ObstacleIdAssignment.returns(
                        partial(
                            lambda observation_0, observation_1, observation_2, ids: (
                                ids(
                                    data.obstacle_ids([3, 2]),
                                    when_observing=observation_0,
                                    and_ids=data.obstacle_ids([]),
                                ),
                                ids(
                                    data.obstacle_ids([2]),
                                    when_observing=observation_1,
                                    and_ids=data.obstacle_ids([2, 3]),
                                ),
                                ids(
                                    data.obstacle_ids([3]),
                                    when_observing=observation_2,
                                    and_ids=data.obstacle_ids([2, 3]),
                                ),
                            ),
                            observation_0 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_03, x_02], shape=(K_0 := 2,)),
                                y=array([y_03, y_02], shape=(K_0,)),
                                heading=array([h_03, h_02], shape=(K_0,)),
                            ),
                            observation_1 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_12], shape=(K_1 := 1,)),
                                y=array([y_12], shape=(K_1,)),
                                heading=array([h_12], shape=(K_1,)),
                            ),
                            observation_2 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_23], shape=(K_2 := 1,)),
                                y=array([y_23], shape=(K_2,)),
                                heading=array([h_23], shape=(K_2,)),
                            ),
                        )
                    ),
                ),
                states_sequence := [observation_0, observation_1, observation_2],
                prediction,
            ),
            (  # Total number of IDs exceeds expected. Older IDs are dropped.
                provider := obstacles.provider.predicting(
                    predictor=stubs.ObstacleMotionPredictor.returns(
                        prediction := data.obstacle_2d_poses(
                            x=np.random.rand(T := 15, K := 3),
                            y=np.random.rand(T, K),
                            heading=np.random.rand(T, K),
                            covariance=np.random.rand(T, 3, 3, K),
                        ),
                        when_history_is=(
                            history := data.obstacle_2d_poses(
                                x=array(
                                    [
                                        # Final tracked IDs: [2, 4, 5] (IDs 3 & 6 dropped)
                                        [np.nan, np.nan, np.nan],
                                        [x_12 := 0.2, np.nan, np.nan],
                                        [x_22 := 0.25, x_24 := 0.4, np.nan],
                                        [np.nan, x_34 := 0.45, x_35 := 0.5],
                                    ],
                                    shape=(H := 4, K := 3),
                                ),
                                y=array(
                                    [
                                        [np.nan, np.nan, np.nan],
                                        [y_12 := 1.2, np.nan, np.nan],
                                        [y_22 := 1.25, y_24 := 1.4, np.nan],
                                        [np.nan, y_34 := 1.45, y_35 := 1.5],
                                    ],
                                    shape=(H, K),
                                ),
                                heading=array(
                                    [
                                        [np.nan, np.nan, np.nan],
                                        [h_12 := 0.1, np.nan, np.nan],
                                        [h_22 := 0.15, h_24 := 0.3, np.nan],
                                        [np.nan, h_34 := 0.35, h_35 := 0.4],
                                    ],
                                    shape=(H, K),
                                ),
                            )
                        ),
                    ),
                    history=types.obstacle_states_running_history.empty(
                        creator=types.obstacle_2d_poses, obstacle_count=K
                    ),
                    id_assignment=stubs.ObstacleIdAssignment.returns(
                        partial(
                            lambda observation_0, history_0, ids_0, observation_1, history_1, ids_1, observation_2, history_2, ids_2, observation_3, history_3, ids_3, ids: (
                                ids(
                                    data.obstacle_ids([3, 6]),
                                    when_observing=observation_0,
                                    and_history=history_0,
                                    and_ids=ids_0,
                                ),
                                ids(
                                    data.obstacle_ids([6, 2, 3]),
                                    when_observing=observation_1,
                                    and_history=history_1,
                                    and_ids=ids_1,
                                ),
                                ids(
                                    data.obstacle_ids([4, 2]),
                                    when_observing=observation_2,
                                    and_history=history_2,
                                    and_ids=ids_2,
                                ),
                                ids(
                                    data.obstacle_ids([5, 4]),
                                    when_observing=observation_3,
                                    and_history=history_3,
                                    and_ids=ids_3,
                                ),
                            ),
                            observation_0 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_03 := 0.2, x_06 := 0.1], shape=(K_0 := 2,)),
                                y=array([y_03 := 0.4, y_06 := 0.5], shape=(K_0,)),
                                heading=array([h_03 := 0.6, h_06 := 0.7], shape=(K_0,)),
                            ),
                            history_0 := data.obstacle_2d_poses(
                                x=np.empty((0, K)),
                                y=np.empty((0, K)),
                                heading=np.empty((0, K)),
                            ),
                            ids_0 := data.obstacle_ids([]),
                            observation_1 := data.obstacle_2d_poses_for_time_step(
                                x=array(
                                    [x_16 := 0.1, x_12, x_13 := 0.3], shape=(K_1 := 3,)
                                ),
                                y=array([y_16 := 1.1, y_12, y_13 := 1.3], shape=(K_1,)),
                                heading=array(
                                    [h_16 := 2.1, h_12, h_13 := 0.2], shape=(K_1,)
                                ),
                            ),
                            history_1 := data.obstacle_2d_poses(
                                x=array([[x_03, x_06, np.nan]], shape=(1, K)),
                                y=array([[y_03, y_06, np.nan]], shape=(1, K)),
                                heading=array([[h_03, h_06, np.nan]], shape=(1, K)),
                            ),
                            ids_1 := data.obstacle_ids([3, 6]),
                            observation_2 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_24, x_22], shape=(K_2 := 2,)),
                                y=array([y_24, y_22], shape=(K_2,)),
                                heading=array([h_24, h_22], shape=(K_2,)),
                            ),
                            history_2 := data.obstacle_2d_poses(
                                x=array(
                                    [[np.nan, x_03, x_06], [x_12, x_13, x_16]],
                                    shape=(2, K),
                                ),
                                y=array(
                                    [[np.nan, y_03, y_06], [y_12, y_13, y_16]],
                                    shape=(2, K),
                                ),
                                heading=array(
                                    [[np.nan, h_03, h_06], [h_12, h_13, h_16]],
                                    shape=(2, K),
                                ),
                            ),
                            ids_2 := data.obstacle_ids([2, 3, 6]),
                            observation_3 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_35, x_34], shape=(K_3 := 2,)),
                                y=array([y_35, y_34], shape=(K_3,)),
                                heading=array([h_35, h_34], shape=(K_3,)),
                            ),
                            history_3 := data.obstacle_2d_poses(
                                x=array(
                                    [
                                        [np.nan, np.nan, x_06],
                                        [x_12, np.nan, x_16],
                                        [x_22, x_24, np.nan],
                                    ],
                                    shape=(3, K),
                                ),
                                y=array(
                                    [
                                        [np.nan, np.nan, y_06],
                                        [y_12, np.nan, y_16],
                                        [y_22, y_24, np.nan],
                                    ],
                                    shape=(3, K),
                                ),
                                heading=array(
                                    [
                                        [np.nan, np.nan, h_06],
                                        [h_12, np.nan, h_16],
                                        [h_22, h_24, np.nan],
                                    ],
                                    shape=(3, K),
                                ),
                            ),
                            ids_3 := data.obstacle_ids([2, 4, 6]),
                        )
                    ),
                ),
                states_sequence := [
                    observation_0,
                    observation_1,
                    observation_2,
                    observation_3,
                ],
                prediction,
            ),
            (  # No limit on total number of IDs.
                provider := obstacles.provider.predicting(
                    predictor=stubs.ObstacleMotionPredictor.returns(
                        prediction := data.obstacle_2d_poses(
                            x=np.random.rand(T := 15, K := 4),
                            y=np.random.rand(T, K),
                            heading=np.random.rand(T, K),
                            covariance=np.random.rand(T, 3, 3, K),
                        ),
                        when_history_is=(
                            history := data.obstacle_2d_poses(
                                x=array(
                                    [
                                        # All 4 IDs should be tracked: [2, 3, 4, 5]
                                        [x_02 := 0.2, x_03 := 0.3, np.nan, np.nan],
                                        [np.nan, np.nan, x_14 := 0.4, x_15 := 0.5],
                                    ],
                                    shape=(H := 2, K := 4),
                                ),
                                y=array(
                                    [
                                        [y_02 := 1.2, y_03 := 1.3, np.nan, np.nan],
                                        [np.nan, np.nan, y_14 := 1.4, y_15 := 1.5],
                                    ],
                                    shape=(H, K),
                                ),
                                heading=array(
                                    [
                                        [h_02 := 0.1, h_03 := 0.2, np.nan, np.nan],
                                        [np.nan, np.nan, h_14 := 0.3, h_15 := 0.4],
                                    ],
                                    shape=(H, K),
                                ),
                            )
                        ),
                    ),
                    # No obstacle count (limit) specified.
                    history=types.obstacle_states_running_history.empty(
                        creator=types.obstacle_2d_poses
                    ),
                    id_assignment=stubs.ObstacleIdAssignment.returns(
                        partial(
                            lambda history_0, history_1, ids: (
                                ids(
                                    data.obstacle_ids([2, 3]),
                                    when_observing=history_0,
                                    and_ids=data.obstacle_ids([]),
                                ),
                                ids(
                                    data.obstacle_ids([4, 5]),
                                    when_observing=history_1,
                                    and_ids=data.obstacle_ids([2, 3]),
                                ),
                            ),
                            history_0 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_02, x_03], shape=(2,)),
                                y=array([y_02, y_03], shape=(2,)),
                                heading=array([h_02, h_03], shape=(2,)),
                            ),
                            history_1 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_14, x_15], shape=(2,)),
                                y=array([y_14, y_15], shape=(2,)),
                                heading=array([h_14, h_15], shape=(2,)),
                            ),
                        )
                    ),
                ),
                states_sequence := [history_0, history_1],
                prediction,
            ),
            (  # Not enough history to fill horizon.
                provider := obstacles.provider.predicting(
                    predictor=stubs.ObstacleMotionPredictor.returns(
                        prediction := data.obstacle_2d_poses(
                            x=np.random.rand(T := 15, K := 2),
                            y=np.random.rand(T, K),
                            heading=np.random.rand(T, K),
                            covariance=np.random.rand(T, 3, 3, K),
                        ),
                        when_history_is=(
                            history := data.obstacle_2d_poses(
                                x=array(
                                    [
                                        # Fixed horizon is 4, but only 2 observations made.
                                        [np.nan, np.nan],
                                        [np.nan, np.nan],
                                        # Data will be shifted towards higher horizon indices.
                                        [x_02 := 0.2, x_03 := 0.3],
                                        [x_12 := 0.25, x_13 := 0.35],
                                    ],
                                    shape=(H := 4, K := 2),
                                ),
                                y=array(
                                    [
                                        [np.nan, np.nan],
                                        [np.nan, np.nan],
                                        [y_02 := 1.2, y_03 := 1.3],
                                        [y_12 := 1.25, y_13 := 1.35],
                                    ],
                                    shape=(H, K),
                                ),
                                heading=array(
                                    [
                                        [np.nan, np.nan],
                                        [np.nan, np.nan],
                                        [h_02 := 0.1, h_03 := 0.2],
                                        [h_12 := 0.15, h_13 := 0.25],
                                    ],
                                    shape=(H, K),
                                ),
                            )
                        ),
                    ),
                    history=types.obstacle_states_running_history.empty(
                        creator=types.obstacle_2d_poses, horizon=H, obstacle_count=K
                    ),
                    id_assignment=stubs.ObstacleIdAssignment.returns(
                        partial(
                            lambda observation_0, history_0, ids_0, observation_1, history_1, ids_1, ids: (
                                ids(
                                    data.obstacle_ids([2, 3]),
                                    when_observing=observation_0,
                                    and_history=history_0,
                                    and_ids=ids_0,
                                ),
                                ids(
                                    data.obstacle_ids([2, 3]),
                                    when_observing=observation_1,
                                    and_history=history_1,
                                    and_ids=ids_1,
                                ),
                            ),
                            observation_0 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_02, x_03], shape=(K,)),
                                y=array([y_02, y_03], shape=(K,)),
                                heading=array([h_02, h_03], shape=(K,)),
                            ),
                            history_0 := data.obstacle_2d_poses(
                                x=array(np.full((H, K), np.nan), shape=(H, K)),
                                y=array(np.full((H, K), np.nan), shape=(H, K)),
                                heading=array(np.full((H, K), np.nan), shape=(H, K)),
                            ),
                            ids_0 := data.obstacle_ids([]),
                            observation_1 := data.obstacle_2d_poses_for_time_step(
                                x=array([x_12, x_13], shape=(K,)),
                                y=array([y_12, y_13], shape=(K,)),
                                heading=array([h_12, h_13], shape=(K,)),
                            ),
                            history_1 := data.obstacle_2d_poses(
                                x=array(
                                    [
                                        [np.nan, np.nan],
                                        [np.nan, np.nan],
                                        [np.nan, np.nan],
                                        [x_02, x_03],
                                    ],
                                    shape=(H, K),
                                ),
                                y=array(
                                    [
                                        [np.nan, np.nan],
                                        [np.nan, np.nan],
                                        [np.nan, np.nan],
                                        [y_02, y_03],
                                    ],
                                    shape=(H, K),
                                ),
                                heading=array(
                                    [
                                        [np.nan, np.nan],
                                        [np.nan, np.nan],
                                        [np.nan, np.nan],
                                        [h_02, h_03],
                                    ],
                                    shape=(H, K),
                                ),
                            ),
                            ids_1 := data.obstacle_ids([2, 3]),
                        )
                    ),
                ),
                states_sequence := [observation_0, observation_1],
                prediction,
            ),
        ]

    @mark.parametrize(
        ["provider", "states_sequence", "expected"],
        [
            *cases(obstacles=obstacles.numpy, data=data.numpy, types=types.numpy),
            *cases(obstacles=obstacles.jax, data=data.jax, types=types.jax),
        ],
    )
    def test[ObstacleStatesForTimeStepT, PredictionT: ObstacleStates](
        self,
        provider: PredictingObstacleStateProvider[
            ObstacleStatesForTimeStepT, Any, Any, PredictionT
        ],
        states_sequence: Sequence[ObstacleStatesForTimeStepT],
        expected: PredictionT,
    ) -> None:
        for states in states_sequence:
            provider.observe(states)

        assert np.allclose(provider(), expected)


class test_that_running_history_tracks_active_ids_when_obstacle_count_is_exceeded:
    @staticmethod
    def cases(types, data) -> Sequence[tuple]:
        return [
            (  # Single persistent obstacle among many transient ones
                history := types.obstacle_states_running_history.empty(
                    creator=types.obstacle_2d_poses, obstacle_count=2
                )
                .append(
                    data.obstacle_2d_poses_for_time_step(
                        x=array([5.0, 100.0], shape=(2,)),
                        y=array([5.0, 0.0], shape=(2,)),
                        heading=array([0.0, 0.0], shape=(2,)),
                    ),
                    ids=data.obstacle_ids([10, 20]),
                )
                .append(
                    data.obstacle_2d_poses_for_time_step(
                        x=array([5.1, 200.0], shape=(2,)),
                        y=array([5.1, 0.0], shape=(2,)),
                        heading=array([0.0, 0.0], shape=(2,)),
                    ),
                    ids=data.obstacle_ids([10, 30]),  # 20 disappears, 30 appears
                )
                .append(
                    data.obstacle_2d_poses_for_time_step(
                        x=array([5.2, 300.0], shape=(2,)),
                        y=array([5.2, 0.0], shape=(2,)),
                        heading=array([0.0, 0.0], shape=(2,)),
                    ),
                    ids=data.obstacle_ids([10, 40]),  # 30 disappears, 40 appears
                )
                .append(
                    data.obstacle_2d_poses_for_time_step(
                        x=array([5.3, 400.0], shape=(2,)),
                        y=array([5.3, 0.0], shape=(2,)),
                        heading=array([0.0, 0.0], shape=(2,)),
                    ),
                    ids=data.obstacle_ids([10, 50]),  # 40 disappears, 50 appears
                ),
                # ID 10 present in all 4 frames, should be retained
                expected_ids := data.obstacle_ids([10, 50]),
            ),
            (  # Multiple persistent IDs among transient ones
                history := types.obstacle_states_running_history.empty(
                    creator=types.obstacle_2d_poses, obstacle_count=3
                )
                .append(
                    data.obstacle_2d_poses_for_time_step(
                        x=array([0.0, 10.0, 20.0], shape=(3,)),
                        y=array([0.0, 0.0, 0.0], shape=(3,)),
                        heading=array([0.0, 0.0, 0.0], shape=(3,)),
                    ),
                    ids=data.obstacle_ids([1, 2, 3]),
                )
                .append(
                    data.obstacle_2d_poses_for_time_step(
                        x=array([0.1, 10.1, 30.0], shape=(3,)),
                        y=array([0.0, 0.0, 0.0], shape=(3,)),
                        heading=array([0.0, 0.0, 0.0], shape=(3,)),
                    ),
                    ids=data.obstacle_ids([1, 2, 4]),  # 3 disappears, 4 appears
                )
                .append(
                    data.obstacle_2d_poses_for_time_step(
                        x=array([0.2, 10.2, 40.0], shape=(3,)),
                        y=array([0.0, 0.0, 0.0], shape=(3,)),
                        heading=array([0.0, 0.0, 0.0], shape=(3,)),
                    ),
                    ids=data.obstacle_ids([1, 2, 5]),  # 4 disappears, 5 appears
                ),
                # IDs 1 and 2 present in all frames, 5 is newest
                expected_ids := data.obstacle_ids([1, 2, 5]),
            ),
        ]

    @mark.parametrize(
        ["history", "expected_ids"],
        [
            *cases(types=types.numpy, data=data.numpy),
            *cases(types=types.jax, data=data.jax),
        ],
    )
    def test(
        self,
        history: ObstacleStatesRunningHistory[Any, ObstacleIds, Any],
        expected_ids: ObstacleIds,
    ) -> None:
        actual_ids = history.ids()
        assert np.allclose(actual_ids, expected_ids)
