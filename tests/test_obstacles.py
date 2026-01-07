from typing import Any, Sequence
from functools import partial

from trajax import PredictingObstacleStateProvider, ObstacleStates, types, obstacles

from numtypes import array

import numpy as np
import jax.numpy as jnp

from tests.dsl import stubs, mppi as data, clear_type
from pytest import mark


@mark.parametrize(
    ["provider", "states_sequence", "expected"],
    [
        (
            provider := obstacles.predicting(
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
                history=types.numpy.obstacle_states_running_history.empty(),
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
                provider := obstacles.predicting(
                    predictor=stubs.ObstacleMotionPredictor.returns(
                        prediction := data.numpy.obstacle_states(
                            x=np.random.rand(T := 10, K := 3),
                            y=np.random.rand(T, K),
                            heading=np.random.rand(T, K),
                            covariance=np.random.rand(T, 3, 3, K),
                        ),
                        when_history_is=(
                            history := data.numpy.obstacle_states(
                                x=np.full(output_shape, np.nan),
                                y=np.full(output_shape, np.nan),
                                heading=np.full(output_shape, np.nan),
                            )
                        ),
                    ),
                    history=types.numpy.obstacle_states_running_history.empty(
                        horizon=horizon, obstacle_count=obstacle_count
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
            provider := obstacles.predicting(
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
                history=types.numpy.obstacle_states_running_history.empty(
                    horizon=H,
                ),
            ),
            states_sequence := [
                *[
                    data.numpy.obstacle_states_for_time_step(
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
        (  # Analogous tests for JAX
            provider := obstacles.predicting(
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
                history=types.jax.obstacle_states_running_history.empty(),
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
            (
                provider := obstacles.predicting(
                    predictor=stubs.ObstacleMotionPredictor.returns(
                        prediction := data.jax.obstacle_states(
                            x=np.random.rand(T := 10, K := 3),
                            y=np.random.rand(T, K),
                            heading=np.random.rand(T, K),
                            covariance=np.random.rand(T, 3, 3, K),
                        ),
                        when_history_is=(
                            history := data.jax.obstacle_states(
                                x=jnp.full(output_shape, jnp.nan),
                                y=jnp.full(output_shape, jnp.nan),
                                heading=jnp.full(output_shape, jnp.nan),
                            )
                        ),
                    ),
                    history=types.jax.obstacle_states_running_history.empty(
                        horizon=horizon, obstacle_count=obstacle_count
                    ),
                ),
                states_sequence := [],
                prediction,
            )
            for horizon, obstacle_count, output_shape in [
                (None, None, (0, 0)),
                (4, None, (4, 0)),
                (None, 5, (0, 5)),
                (6, 3, (6, 3)),
            ]
        ],
        (  # Old observations should be dropped when horizon is exceeded
            provider := obstacles.predicting(
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
                history=types.jax.obstacle_states_running_history.empty(
                    horizon=H,
                ),
            ),
            states_sequence := [
                *[
                    data.jax.obstacle_states_for_time_step(
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
    ],
)
def test_that_obstacle_state_provider_provides_forecasts_from_obstacle_motion_predictor[
    ObstacleStatesForTimeStepT,
    PredictionT: ObstacleStates,
](
    provider: PredictingObstacleStateProvider[
        ObstacleStatesForTimeStepT, Any, Any, PredictionT
    ],
    states_sequence: Sequence[ObstacleStatesForTimeStepT],
    expected: PredictionT,
) -> None:
    for states in states_sequence:
        provider.observe(states)

    assert np.allclose(provider(), expected)


T = clear_type
H = clear_type
K = clear_type


@mark.parametrize(
    ["provider", "states_sequence", "expected"],
    [
        (
            provider := obstacles.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.numpy.obstacle_states(
                        x=np.random.rand(T := 15, K := 3),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.numpy.obstacle_states(
                            x=(
                                x := array(
                                    [
                                        x_0 := [x_02 := 0.1, x_03 := 0.3, x_06 := 0.5],
                                        x_1 := [x_12 := 0.9, x_13 := 0.2, x_16 := 0.7],
                                        x_2 := [x_22 := 0.8, x_23 := 0.4, x_26 := 0.6],
                                        x_3 := [x_32 := 0.6, x_33 := 0.2, x_36 := 0.1],
                                    ],
                                    shape=(H := 4, K := 3),
                                )
                            ),
                            y=(
                                y := array(
                                    [
                                        y_0 := [y_02 := 1.2, y_03 := 4.3, y_06 := 1.2],
                                        y_1 := [y_12 := 1.5, y_13 := 0.8, y_16 := 1.9],
                                        y_2 := [y_22 := 1.3, y_23 := 2.4, y_26 := 1.4],
                                        y_3 := [y_32 := 1.7, y_33 := 1.2, y_36 := 1.6],
                                    ],
                                    shape=(H, K),
                                )
                            ),
                            heading=(
                                heading := array(
                                    [
                                        h_0 := [h_02 := 2.1, h_03 := 0.8, h_06 := 0.3],
                                        h_1 := [h_12 := 0.4, h_13 := 4.2, h_16 := 4.3],
                                        h_2 := [h_22 := 1.2, h_23 := 0.3, h_26 := 0.5],
                                        h_3 := [h_32 := 5.2, h_33 := 0.4, h_36 := 1.4],
                                    ],
                                    shape=(H, K),
                                )
                            ),
                        )
                    ),
                ),
                history=types.numpy.obstacle_states_running_history.empty(),
                id_assignment=stubs.ObstacleIdAssignment.returns(
                    partial(
                        lambda history_0,
                        history_1,
                        history_2,
                        history_3,
                        x,
                        y,
                        heading,
                        ids: (
                            # IDs don't start from 0 intentionally.
                            ids(
                                data.numpy.obstacle_ids([6, 2, 3]),
                                when_observing=history_0,
                                with_history=data.numpy.obstacle_states(
                                    x=np.empty((0, 0)),
                                    y=np.empty((0, 0)),
                                    heading=np.empty((0, 0)),
                                ),
                            ),
                            ids(
                                data.numpy.obstacle_ids([3, 6, 2]),
                                when_observing=history_1,
                                with_history=data.numpy.obstacle_states(
                                    x=x[:1], y=y[:1], heading=heading[:1]
                                ),
                            ),
                            ids(
                                data.numpy.obstacle_ids([2, 3, 6]),
                                when_observing=history_2,
                                with_history=data.numpy.obstacle_states(
                                    x=x[:2], y=y[:2], heading=heading[:2]
                                ),
                            ),
                            ids(
                                data.numpy.obstacle_ids([3, 2, 6]),
                                when_observing=history_3,
                                with_history=data.numpy.obstacle_states(
                                    x=x[:3], y=y[:3], heading=heading[:3]
                                ),
                            ),
                        ),
                        history_0 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_06, x_02, x_03], shape=(K,)),
                            y=array([y_06, y_02, y_03], shape=(K,)),
                            heading=array([h_06, h_02, h_03], shape=(K,)),
                        ),
                        history_1 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_13, x_16, x_12], shape=(K,)),
                            y=array([y_13, y_16, y_12], shape=(K,)),
                            heading=array([h_13, h_16, h_12], shape=(K,)),
                        ),
                        history_2 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_22, x_23, x_26], shape=(K,)),
                            y=array([y_22, y_23, y_26], shape=(K,)),
                            heading=array([h_22, h_23, h_26], shape=(K,)),
                        ),
                        history_3 := data.numpy.obstacle_states_for_time_step(
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
            states_sequence := [history_0, history_1, history_2, history_3],
            prediction,
        ),
        (  # Obstacle disappears then reappears
            provider := obstacles.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.numpy.obstacle_states(
                        x=np.random.rand(T := 15, K := 4),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.numpy.obstacle_states(
                            x=array(
                                [
                                    # IDs: (2, 3, 5, 6).
                                    # ID = 3 Disappears
                                    # ID = 5 Appears later
                                    # ID = 6 Disappears then reappears
                                    [x_02 := 0.2, x_03 := 0.3, np.nan, x_06 := 0.6],
                                    [x_12 := 0.2, np.nan, np.nan, np.nan],
                                    [x_22 := 0.3, np.nan, x_25 := 0.6, np.nan],
                                    [x_32 := 0.3, np.nan, x_35 := 0.4, x_36 := 0.2],
                                ],
                                shape=(H := 4, K := 4),
                            ),
                            y=array(
                                [
                                    [y_02 := 1.2, y_03 := 1.3, np.nan, y_06 := 1.6],
                                    [y_12 := 1.2, np.nan, np.nan, np.nan],
                                    [y_22 := 1.3, np.nan, y_25 := 1.6, np.nan],
                                    [y_32 := 1.3, np.nan, y_35 := 1.4, y_36 := 1.2],
                                ],
                                shape=(H, K),
                            ),
                            heading=array(
                                [
                                    [h_02 := 0.1, h_03 := 0.2, np.nan, h_06 := 0.4],
                                    [h_12 := 0.1, np.nan, np.nan, np.nan],
                                    [h_22 := 0.2, np.nan, h_25 := 0.4, np.nan],
                                    [h_32 := 0.2, np.nan, h_35 := 0.3, h_36 := 0.1],
                                ],
                                shape=(H, K),
                            ),
                        )
                    ),
                ),
                history=types.numpy.obstacle_states_running_history.empty(
                    obstacle_count=K
                ),
                id_assignment=stubs.ObstacleIdAssignment.returns(
                    partial(
                        lambda history_0, history_1, history_2, history_3, ids: (
                            ids(
                                data.numpy.obstacle_ids([6, 3, 2]),
                                when_observing=history_0,
                            ),
                            ids(
                                data.numpy.obstacle_ids([2]),
                                when_observing=history_1,
                            ),
                            ids(
                                data.numpy.obstacle_ids([5, 2]),
                                when_observing=history_2,
                            ),
                            ids(
                                data.numpy.obstacle_ids([2, 5, 6]),
                                when_observing=history_3,
                            ),
                        ),
                        history_0 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_06, x_03, x_02], shape=(K_0 := 3,)),
                            y=array([y_06, y_03, y_02], shape=(K_0,)),
                            heading=array([h_06, h_03, h_02], shape=(K_0,)),
                        ),
                        history_1 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_02], shape=(K_1 := 1,)),
                            y=array([y_12], shape=(K_1,)),
                            heading=array([h_12], shape=(K_1,)),
                        ),
                        history_2 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_25, x_22], shape=(K_2 := 2,)),
                            y=array([y_25, y_22], shape=(K_2,)),
                            heading=array([h_25, h_22], shape=(K_2,)),
                        ),
                        history_3 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_32, x_35, x_36], shape=(K_3 := 3,)),
                            y=array([y_32, y_35, y_36], shape=(K_3,)),
                            heading=array([h_32, h_35, h_36], shape=(K_3,)),
                        ),
                    )
                ),
            ),
            states_sequence := [history_0, history_1, history_2, history_3],
            prediction,
        ),
        (  # Total number of IDs is less than expected.
            provider := obstacles.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.numpy.obstacle_states(
                        x=np.random.rand(T := 15, K := 4),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.numpy.obstacle_states(
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
                history=types.numpy.obstacle_states_running_history.empty(
                    obstacle_count=K
                ),
                id_assignment=stubs.ObstacleIdAssignment.returns(
                    partial(
                        lambda history_0, history_1, history_2, ids: (
                            ids(
                                data.numpy.obstacle_ids([3, 2]),
                                when_observing=history_0,
                            ),
                            ids(
                                data.numpy.obstacle_ids([2]),
                                when_observing=history_1,
                            ),
                            ids(
                                data.numpy.obstacle_ids([3]),
                                when_observing=history_2,
                            ),
                        ),
                        history_0 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_03, x_02], shape=(K_0 := 2,)),
                            y=array([y_03, y_02], shape=(K_0,)),
                            heading=array([h_03, h_02], shape=(K_0,)),
                        ),
                        history_1 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_12], shape=(K_1 := 1,)),
                            y=array([y_12], shape=(K_1,)),
                            heading=array([h_12], shape=(K_1,)),
                        ),
                        history_2 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_23], shape=(K_2 := 1,)),
                            y=array([y_23], shape=(K_2,)),
                            heading=array([h_23], shape=(K_2,)),
                        ),
                    )
                ),
            ),
            states_sequence := [history_0, history_1, history_2],
            prediction,
        ),
        (  # Total number of IDs exceeds expected. Older IDs are dropped.
            provider := obstacles.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.numpy.obstacle_states(
                        x=np.random.rand(T := 15, K := 3),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.numpy.obstacle_states(
                            x=array(
                                [
                                    # Final tracked IDs: [2, 4, 5] (IDs 3 & 6 dropped)
                                    [np.nan, np.nan, np.nan],
                                    [x_02 := 0.2, np.nan, np.nan],
                                    [x_12 := 0.25, x_14 := 0.4, np.nan],
                                    [np.nan, x_24 := 0.45, x_25 := 0.5],
                                ],
                                shape=(H := 4, K := 3),
                            ),
                            y=array(
                                [
                                    [np.nan, np.nan, np.nan],
                                    [y_02 := 1.2, np.nan, np.nan],
                                    [y_12 := 1.25, y_14 := 1.4, np.nan],
                                    [np.nan, y_24 := 1.45, y_25 := 1.5],
                                ],
                                shape=(H, K),
                            ),
                            heading=array(
                                [
                                    [np.nan, np.nan, np.nan],
                                    [h_02 := 0.1, np.nan, np.nan],
                                    [h_12 := 0.15, h_14 := 0.3, np.nan],
                                    [np.nan, h_24 := 0.35, h_25 := 0.4],
                                ],
                                shape=(H, K),
                            ),
                        )
                    ),
                ),
                history=types.numpy.obstacle_states_running_history.empty(
                    obstacle_count=K
                ),
                id_assignment=stubs.ObstacleIdAssignment.returns(
                    partial(
                        lambda history_0, history_1, history_2, history_3, ids: (
                            ids(
                                data.numpy.obstacle_ids([3, 6]),
                                when_observing=history_0,
                            ),
                            ids(
                                data.numpy.obstacle_ids([6, 2, 3]),
                                when_observing=history_1,
                            ),
                            ids(
                                data.numpy.obstacle_ids([4, 2]),
                                when_observing=history_2,
                            ),
                            ids(
                                data.numpy.obstacle_ids([5, 4]),
                                when_observing=history_3,
                            ),
                        ),
                        history_0 := data.numpy.obstacle_states_for_time_step(
                            x=array([0.2, 0.1], shape=(2,)),
                            y=array([0.4, 0.5], shape=(2,)),
                            heading=array([0.6, 0.7], shape=(2,)),
                        ),
                        history_1 := data.numpy.obstacle_states_for_time_step(
                            x=array([0.1, x_02, 0.3], shape=(3,)),
                            y=array([1.1, y_02, 1.3], shape=(3,)),
                            heading=array([2.1, h_02, 0.2], shape=(3,)),
                        ),
                        history_2 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_14, x_12], shape=(2,)),
                            y=array([y_14, y_12], shape=(2,)),
                            heading=array([h_14, h_12], shape=(2,)),
                        ),
                        history_3 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_25, x_24], shape=(2,)),
                            y=array([y_25, y_24], shape=(2,)),
                            heading=array([h_25, h_24], shape=(2,)),
                        ),
                    )
                ),
            ),
            states_sequence := [history_0, history_1, history_2, history_3],
            prediction,
        ),
        (  # No limit on total number of IDs.
            provider := obstacles.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.numpy.obstacle_states(
                        x=np.random.rand(T := 15, K := 4),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.numpy.obstacle_states(
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
                history=types.numpy.obstacle_states_running_history.empty(),
                id_assignment=stubs.ObstacleIdAssignment.returns(
                    partial(
                        lambda history_0, history_1, ids: (
                            ids(
                                data.numpy.obstacle_ids([2, 3]),
                                when_observing=history_0,
                            ),
                            ids(
                                data.numpy.obstacle_ids([4, 5]),
                                when_observing=history_1,
                            ),
                        ),
                        history_0 := data.numpy.obstacle_states_for_time_step(
                            x=array([x_02, x_03], shape=(2,)),
                            y=array([y_02, y_03], shape=(2,)),
                            heading=array([h_02, h_03], shape=(2,)),
                        ),
                        history_1 := data.numpy.obstacle_states_for_time_step(
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
        (  # Analogous cases for JAX.
            provider := obstacles.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.jax.obstacle_states(
                        x=np.random.rand(T := 15, K := 3),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.jax.obstacle_states(
                            x=(
                                x := array(
                                    [
                                        x_0 := [x_02 := 0.1, x_03 := 0.3, x_06 := 0.5],
                                        x_1 := [x_12 := 0.9, x_13 := 0.2, x_16 := 0.7],
                                        x_2 := [x_22 := 0.8, x_23 := 0.4, x_26 := 0.6],
                                        x_3 := [x_32 := 0.6, x_33 := 0.2, x_36 := 0.1],
                                    ],
                                    shape=(H := 4, K := 3),
                                )
                            ),
                            y=(
                                y := array(
                                    [
                                        y_0 := [y_02 := 1.2, y_03 := 4.3, y_06 := 1.2],
                                        y_1 := [y_12 := 1.5, y_13 := 0.8, y_16 := 1.9],
                                        y_2 := [y_22 := 1.3, y_23 := 2.4, y_26 := 1.4],
                                        y_3 := [y_32 := 1.7, y_33 := 1.2, y_36 := 1.6],
                                    ],
                                    shape=(H, K),
                                )
                            ),
                            heading=(
                                heading := array(
                                    [
                                        h_0 := [h_02 := 2.1, h_03 := 0.8, h_06 := 0.3],
                                        h_1 := [h_12 := 0.4, h_13 := 4.2, h_16 := 4.3],
                                        h_2 := [h_22 := 1.2, h_23 := 0.3, h_26 := 0.5],
                                        h_3 := [h_32 := 5.2, h_33 := 0.4, h_36 := 1.4],
                                    ],
                                    shape=(H, K),
                                )
                            ),
                        )
                    ),
                ),
                history=types.jax.obstacle_states_running_history.empty(),
                id_assignment=stubs.ObstacleIdAssignment.returns(
                    partial(
                        lambda history_0,
                        history_1,
                        history_2,
                        history_3,
                        x,
                        y,
                        heading,
                        ids: (
                            ids(
                                data.jax.obstacle_ids([6, 2, 3]),
                                when_observing=history_0,
                                with_history=data.jax.obstacle_states(
                                    x=np.empty((0, 0)),
                                    y=np.empty((0, 0)),
                                    heading=np.empty((0, 0)),
                                ),
                            ),
                            ids(
                                data.jax.obstacle_ids([3, 6, 2]),
                                when_observing=history_1,
                                with_history=data.jax.obstacle_states(
                                    x=x[:1], y=y[:1], heading=heading[:1]
                                ),
                            ),
                            ids(
                                data.jax.obstacle_ids([2, 3, 6]),
                                when_observing=history_2,
                                with_history=data.jax.obstacle_states(
                                    x=x[:2], y=y[:2], heading=heading[:2]
                                ),
                            ),
                            ids(
                                data.jax.obstacle_ids([3, 2, 6]),
                                when_observing=history_3,
                                with_history=data.jax.obstacle_states(
                                    x=x[:3], y=y[:3], heading=heading[:3]
                                ),
                            ),
                        ),
                        history_0 := data.jax.obstacle_states_for_time_step(
                            x=array([x_06, x_02, x_03], shape=(K,)),
                            y=array([y_06, y_02, y_03], shape=(K,)),
                            heading=array([h_06, h_02, h_03], shape=(K,)),
                        ),
                        history_1 := data.jax.obstacle_states_for_time_step(
                            x=array([x_13, x_16, x_12], shape=(K,)),
                            y=array([y_13, y_16, y_12], shape=(K,)),
                            heading=array([h_13, h_16, h_12], shape=(K,)),
                        ),
                        history_2 := data.jax.obstacle_states_for_time_step(
                            x=array([x_22, x_23, x_26], shape=(K,)),
                            y=array([y_22, y_23, y_26], shape=(K,)),
                            heading=array([h_22, h_23, h_26], shape=(K,)),
                        ),
                        history_3 := data.jax.obstacle_states_for_time_step(
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
            states_sequence := [history_0, history_1, history_2, history_3],
            prediction,
        ),
        (
            provider := obstacles.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.jax.obstacle_states(
                        x=np.random.rand(T := 15, K := 4),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.jax.obstacle_states(
                            x=array(
                                [
                                    [x_02 := 0.2, x_03 := 0.3, np.nan, x_06 := 0.6],
                                    [x_12 := 0.2, np.nan, np.nan, np.nan],
                                    [x_22 := 0.3, np.nan, x_25 := 0.6, np.nan],
                                    [x_32 := 0.3, np.nan, x_35 := 0.4, x_36 := 0.2],
                                ],
                                shape=(H := 4, K),
                            ),
                            y=array(
                                [
                                    [y_02 := 1.2, y_03 := 1.3, np.nan, y_06 := 1.6],
                                    [y_12 := 1.2, np.nan, np.nan, np.nan],
                                    [y_22 := 1.3, np.nan, y_25 := 1.6, np.nan],
                                    [y_32 := 1.3, np.nan, y_35 := 1.4, y_36 := 1.2],
                                ],
                                shape=(H, K),
                            ),
                            heading=array(
                                [
                                    [h_02 := 0.1, h_03 := 0.2, np.nan, h_06 := 0.4],
                                    [h_12 := 0.1, np.nan, np.nan, np.nan],
                                    [h_22 := 0.2, np.nan, h_25 := 0.4, np.nan],
                                    [h_32 := 0.2, np.nan, h_35 := 0.3, h_36 := 0.1],
                                ],
                                shape=(H, K),
                            ),
                        )
                    ),
                ),
                history=types.jax.obstacle_states_running_history.empty(
                    obstacle_count=K
                ),
                id_assignment=stubs.ObstacleIdAssignment.returns(
                    partial(
                        lambda history_0, history_1, history_2, history_3, ids: (
                            ids(
                                data.jax.obstacle_ids([6, 3, 2]),
                                when_observing=history_0,
                            ),
                            ids(
                                data.jax.obstacle_ids([2]),
                                when_observing=history_1,
                            ),
                            ids(
                                data.jax.obstacle_ids([5, 2]),
                                when_observing=history_2,
                            ),
                            ids(
                                data.jax.obstacle_ids([2, 5, 6]),
                                when_observing=history_3,
                            ),
                        ),
                        history_0 := data.jax.obstacle_states_for_time_step(
                            x=array([x_06, x_03, x_02], shape=(K_0 := 3,)),
                            y=array([y_06, y_03, y_02], shape=(K_0,)),
                            heading=array([h_06, h_03, h_02], shape=(K_0,)),
                        ),
                        history_1 := data.jax.obstacle_states_for_time_step(
                            x=array([x_02], shape=(K_1 := 1,)),
                            y=array([y_12], shape=(K_1,)),
                            heading=array([h_12], shape=(K_1,)),
                        ),
                        history_2 := data.jax.obstacle_states_for_time_step(
                            x=array([x_25, x_22], shape=(K_2 := 2,)),
                            y=array([y_25, y_22], shape=(K_2,)),
                            heading=array([h_25, h_22], shape=(K_2,)),
                        ),
                        history_3 := data.jax.obstacle_states_for_time_step(
                            x=array([x_32, x_35, x_36], shape=(K_3 := 3,)),
                            y=array([y_32, y_35, y_36], shape=(K_3,)),
                            heading=array([h_32, h_35, h_36], shape=(K_3,)),
                        ),
                    )
                ),
            ),
            states_sequence := [history_0, history_1, history_2, history_3],
            prediction,
        ),
        (
            provider := obstacles.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.jax.obstacle_states(
                        x=np.random.rand(T := 15, K := 4),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.jax.obstacle_states(
                            x=array(
                                [
                                    [x_02 := 0.2, x_03 := 0.3, np.nan, np.nan],
                                    [x_12 := 0.1, np.nan, np.nan, np.nan],
                                    [np.nan, x_23 := 0.4, np.nan, np.nan],
                                ],
                                shape=(H := 3, K),
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
                history=types.jax.obstacle_states_running_history.empty(
                    obstacle_count=K
                ),
                id_assignment=stubs.ObstacleIdAssignment.returns(
                    partial(
                        lambda history_0, history_1, history_2, ids: (
                            ids(
                                data.jax.obstacle_ids([3, 2]),
                                when_observing=history_0,
                            ),
                            ids(
                                data.jax.obstacle_ids([2]),
                                when_observing=history_1,
                            ),
                            ids(
                                data.jax.obstacle_ids([3]),
                                when_observing=history_2,
                            ),
                        ),
                        history_0 := data.jax.obstacle_states_for_time_step(
                            x=array([x_03, x_02], shape=(K_0 := 2,)),
                            y=array([y_03, y_02], shape=(K_0,)),
                            heading=array([h_03, h_02], shape=(K_0,)),
                        ),
                        history_1 := data.jax.obstacle_states_for_time_step(
                            x=array([x_12], shape=(K_1 := 1,)),
                            y=array([y_12], shape=(K_1,)),
                            heading=array([h_12], shape=(K_1,)),
                        ),
                        history_2 := data.jax.obstacle_states_for_time_step(
                            x=array([x_23], shape=(K_2 := 1,)),
                            y=array([y_23], shape=(K_2,)),
                            heading=array([h_23], shape=(K_2,)),
                        ),
                    )
                ),
            ),
            states_sequence := [history_0, history_1, history_2],
            prediction,
        ),
        (
            provider := obstacles.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.jax.obstacle_states(
                        x=np.random.rand(T := 15, K := 3),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.jax.obstacle_states(
                            x=array(
                                [
                                    [np.nan, np.nan, np.nan],
                                    [x_02 := 0.2, np.nan, np.nan],
                                    [x_12 := 0.25, x_14 := 0.4, np.nan],
                                    [np.nan, x_24 := 0.45, x_25 := 0.5],
                                ],
                                shape=(H := 4, K),
                            ),
                            y=array(
                                [
                                    [np.nan, np.nan, np.nan],
                                    [y_02 := 1.2, np.nan, np.nan],
                                    [y_12 := 1.25, y_14 := 1.4, np.nan],
                                    [np.nan, y_24 := 1.45, y_25 := 1.5],
                                ],
                                shape=(H, K),
                            ),
                            heading=array(
                                [
                                    [np.nan, np.nan, np.nan],
                                    [h_02 := 0.1, np.nan, np.nan],
                                    [h_12 := 0.15, h_14 := 0.3, np.nan],
                                    [np.nan, h_24 := 0.35, h_25 := 0.4],
                                ],
                                shape=(H, K),
                            ),
                        )
                    ),
                ),
                history=types.jax.obstacle_states_running_history.empty(
                    obstacle_count=K
                ),
                id_assignment=stubs.ObstacleIdAssignment.returns(
                    partial(
                        lambda history_0, history_1, history_2, history_3, ids: (
                            ids(
                                data.jax.obstacle_ids([3, 6]),
                                when_observing=history_0,
                            ),
                            ids(
                                data.jax.obstacle_ids([6, 2, 3]),
                                when_observing=history_1,
                            ),
                            ids(
                                data.jax.obstacle_ids([4, 2]),
                                when_observing=history_2,
                            ),
                            ids(
                                data.jax.obstacle_ids([5, 4]),
                                when_observing=history_3,
                            ),
                        ),
                        history_0 := data.jax.obstacle_states_for_time_step(
                            x=array([0.2, 0.1], shape=(2,)),
                            y=array([0.4, 0.5], shape=(2,)),
                            heading=array([0.6, 0.7], shape=(2,)),
                        ),
                        history_1 := data.jax.obstacle_states_for_time_step(
                            x=array([0.1, x_02, 0.3], shape=(3,)),
                            y=array([1.1, y_02, 1.3], shape=(3,)),
                            heading=array([2.1, h_02, 0.2], shape=(3,)),
                        ),
                        history_2 := data.jax.obstacle_states_for_time_step(
                            x=array([x_14, x_12], shape=(2,)),
                            y=array([y_14, y_12], shape=(2,)),
                            heading=array([h_14, h_12], shape=(2,)),
                        ),
                        history_3 := data.jax.obstacle_states_for_time_step(
                            x=array([x_25, x_24], shape=(2,)),
                            y=array([y_25, y_24], shape=(2,)),
                            heading=array([h_25, h_24], shape=(2,)),
                        ),
                    )
                ),
            ),
            states_sequence := [history_0, history_1, history_2, history_3],
            prediction,
        ),
        (
            provider := obstacles.predicting(
                predictor=stubs.ObstacleMotionPredictor.returns(
                    prediction := data.jax.obstacle_states(
                        x=np.random.rand(T := 15, K := 4),
                        y=np.random.rand(T, K),
                        heading=np.random.rand(T, K),
                        covariance=np.random.rand(T, 3, 3, K),
                    ),
                    when_history_is=(
                        history := data.jax.obstacle_states(
                            x=array(
                                [
                                    [x_02 := 0.2, x_03 := 0.3, np.nan, np.nan],
                                    [np.nan, np.nan, x_14 := 0.4, x_15 := 0.5],
                                ],
                                shape=(H := 2, K),
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
                history=types.jax.obstacle_states_running_history.empty(),
                id_assignment=stubs.ObstacleIdAssignment.returns(
                    partial(
                        lambda history_0, history_1, ids: (
                            ids(
                                data.jax.obstacle_ids([2, 3]),
                                when_observing=history_0,
                            ),
                            ids(
                                data.jax.obstacle_ids([4, 5]),
                                when_observing=history_1,
                            ),
                        ),
                        history_0 := data.jax.obstacle_states_for_time_step(
                            x=array([x_02, x_03], shape=(2,)),
                            y=array([y_02, y_03], shape=(2,)),
                            heading=array([h_02, h_03], shape=(2,)),
                        ),
                        history_1 := data.jax.obstacle_states_for_time_step(
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
    ],
)
def test_that_obstacle_state_provider_uses_specified_id_assignment[
    ObstacleStatesForTimeStepT,
    PredictionT: ObstacleStates,
](
    provider: PredictingObstacleStateProvider[
        ObstacleStatesForTimeStepT, Any, Any, PredictionT
    ],
    states_sequence: Sequence[ObstacleStatesForTimeStepT],
    expected: PredictionT,
) -> None:
    for states in states_sequence:
        provider.observe(states)

    assert np.allclose(provider(), expected)
