from typing import Sequence, Callable

from trajax import (
    MetricRegistry,
    CollisionMetric,
    Mppi,
    ObstacleStateObserver,
    collectors,
    metrics,
    types,
)

from numtypes import Array, BoolArray, array

import numpy as np

from tests.dsl import stubs, mppi as data
from pytest import mark
from pytest_subtests import SubTests


class test_that_collision_is_detected_when_distance_is_below_threshold:
    @staticmethod
    def cases(data, types) -> Sequence[tuple]:
        return [
            (
                registry := metrics.registry(
                    metric := metrics.collision(
                        distance_threshold=0.5,
                        distance=stubs.DistanceExtractor.returns(
                            data.distance(
                                array(
                                    [
                                        [[[1.0]], [[2.0]], [[3.0]]],  # t=0
                                        [[[0.4]], [[0.6]], [[0.8]]],  # t=1
                                        [[[2.0]], [[2.0]], [[-0.1]]],  # t=2
                                        [[[0.3]], [[0.2]], [[0.1]]],  # t=3
                                    ],
                                    shape=(T := 4, V := 3, M := 1, N := 1),
                                )
                            ),
                            when_states_are=(
                                states := data.state_batch(
                                    np.random.rand(T, D_x := 5, M),
                                )
                            ),
                            and_obstacle_states_are=(
                                obstacle_states := data.obstacle_state_samples(
                                    x=np.random.rand(T, K := 5, N),
                                    y=np.random.rand(T, K, N),
                                )
                            ),
                        ),
                    ),
                    collectors=collectors.registry(
                        mppi := collectors.states.decorating(
                            stubs.Mppi.create(),
                            transformer=types.simple.state_sequence.of_states,
                        ),
                        observer := collectors.obstacle_states.decorating(
                            stubs.ObstacleStateObserver.create(),
                            transformer=types.obstacle_states.of_states,
                        ),
                    ),
                ),
                metric,
                mppi,
                observer,
                horizon := T,
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                states_at := lambda t, states=states: states.at(time_step=t, rollout=0),
                obstacle_states_at := (
                    lambda t, obstacle_states=obstacle_states: obstacle_states.at(
                        time_step=t, sample=0
                    )
                ),
                expected_distances := array(
                    [
                        [1.0, 2.0, 3.0],  # t=0
                        [0.4, 0.6, 0.8],  # t=1
                        [2.0, 2.0, -0.1],  # t=2
                        [0.3, 0.2, 0.1],  # t=3
                    ],
                    shape=(T, V),
                ),
                expected_min_distances := array([0.3, 0.2, -0.1], shape=(V,)),
                expected_collisions := array(
                    [
                        [False, False, False],  # t=0
                        [True, False, False],  # t=1
                        [False, False, True],  # t=2
                        [True, True, True],  # t=3
                    ],
                    shape=(T, V),
                ),
                expected_collision_detected := True,
            )
        ]

    @mark.parametrize(
        [
            "registry",
            "metric",
            "mppi",
            "observer",
            "horizon",
            "nominal_input",
            "states_at",
            "obstacle_states_at",
            "expected_distances",
            "expected_min_distances",
            "expected_collisions",
            "expected_collision_detected",
        ],
        [
            *cases(data=data.numpy, types=types.numpy),
            *cases(data=data.jax, types=types.jax),
        ],
    )
    def test[StateT, InputSequenceT, ObstacleStatesForTimeStepT](
        self,
        registry: MetricRegistry,
        metric: CollisionMetric,
        mppi: Mppi[StateT, InputSequenceT],
        observer: ObstacleStateObserver[ObstacleStatesForTimeStepT],
        horizon: int,
        nominal_input: InputSequenceT,
        states_at: Callable[[int], StateT],
        obstacle_states_at: Callable[[int], ObstacleStatesForTimeStepT],
        expected_distances: Array,
        expected_min_distances: Array,
        expected_collisions: BoolArray,
        expected_collision_detected: bool,
        subtests: SubTests,
    ) -> None:
        for step in range(horizon):
            mppi.step(
                temperature=1.0,
                nominal_input=nominal_input,
                initial_state=states_at(step),
            )

            observer.observe(obstacle_states_at(step))

        # Metrics can be queried by instance (type safe) or by name (no type info).
        with subtests.test("by instance"):
            results = registry.get(metric)

            assert np.allclose(results.distances, expected_distances)
            assert np.allclose(results.min_distances, expected_min_distances)
            assert np.allclose(results.collisions, expected_collisions)
            assert results.collision_detected == expected_collision_detected

        with subtests.test("by name"):
            results = registry.get(metric.name)

            assert np.allclose(results.distances, expected_distances)
            assert np.allclose(results.min_distances, expected_min_distances)
            assert np.allclose(results.collisions, expected_collisions)
            assert results.collision_detected == expected_collision_detected


class test_that_metrics_are_recomputed_when_new_data_is_collected:
    @staticmethod
    def cases(data, types) -> Sequence[tuple]:
        return [
            (
                registry := metrics.registry(
                    metric := metrics.collision(
                        distance_threshold=0.5,
                        distance=stubs.DistanceExtractor.returns(
                            # Distances for final call.
                            data.distance(
                                distance_data := array(
                                    [
                                        [[[1.0]], [[2.0]], [[3.0]]],  # t=0
                                        [[[0.4]], [[0.6]], [[0.8]]],  # t=1
                                        [[[2.0]], [[2.0]], [[-0.1]]],  # t=2
                                        [[[0.3]], [[0.2]], [[0.1]]],  # t=3
                                    ],
                                    shape=(T := 4, V := 3, M := 1, N := 1),
                                )
                            ),
                            when_states_are=(
                                states := data.state_batch(
                                    states_data := np.random.rand(T, D_x := 5, M),
                                )
                            ),
                            and_obstacle_states_are=(
                                obstacle_states := data.obstacle_state_samples(
                                    x=(
                                        obstacles_x_data := np.random.rand(T, K := 5, N)
                                    ),
                                    y=(obstacles_y_data := np.random.rand(T, K, N)),
                                )
                            ),
                        ).or_returns(
                            # Distances for initial call.
                            data.distance(distance_data[:-1]),
                            when_states_are=data.state_batch(states_data[:-1]),
                            and_obstacle_states_are=data.obstacle_state_samples(
                                x=obstacles_x_data[:-1],
                                y=obstacles_y_data[:-1],
                            ),
                        ),
                    ),
                    collectors=collectors.registry(
                        mppi := collectors.states.decorating(
                            stubs.Mppi.create(),
                            transformer=types.simple.state_sequence.of_states,
                        ),
                        observer := collectors.obstacle_states.decorating(
                            stubs.ObstacleStateObserver.create(),
                            transformer=types.obstacle_states.of_states,
                        ),
                    ),
                ),
                metric,
                mppi,
                observer,
                horizon := T,
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                states_at := lambda t: states.at(time_step=t, rollout=0),
                obstacle_states_at := lambda t: obstacle_states.at(
                    time_step=t, sample=0
                ),
                expected_initial_collisions := array(
                    [
                        [False, False, False],  # t=0
                        [True, False, False],  # t=1
                        [False, False, True],  # t=2
                    ],
                    shape=(T - 1, V),
                ),
                expected_final_collisions := array(
                    [
                        [False, False, False],  # t=0
                        [True, False, False],  # t=1
                        [False, False, True],  # t=2
                        [True, True, True],  # t=3
                    ],
                    shape=(T, V),
                ),
            )
        ]

    @mark.parametrize(
        [
            "registry",
            "metric",
            "mppi",
            "observer",
            "horizon",
            "nominal_input",
            "states_at",
            "obstacle_states_at",
            "expected_initial_collisions",
            "expected_final_collisions",
        ],
        [
            *cases(data=data.numpy, types=types.numpy),
            *cases(data=data.jax, types=types.jax),
        ],
    )
    def test[StateT, InputSequenceT, ObstacleStatesForTimeStepT](
        self,
        registry: MetricRegistry,
        metric: CollisionMetric,
        mppi: Mppi[StateT, InputSequenceT],
        observer: ObstacleStateObserver[ObstacleStatesForTimeStepT],
        horizon: int,
        nominal_input: InputSequenceT,
        states_at: Callable[[int], StateT],
        obstacle_states_at: Callable[[int], ObstacleStatesForTimeStepT],
        expected_initial_collisions: BoolArray,
        expected_final_collisions: BoolArray,
    ) -> None:
        for step in range(horizon - 1):
            mppi.step(
                temperature=1.0,
                nominal_input=nominal_input,
                initial_state=states_at(step),
            )

            observer.observe(obstacle_states_at(step))

        assert np.allclose(registry.get(metric).collisions, expected_initial_collisions)

        # Collect new data.
        mppi.step(
            temperature=1.0,
            nominal_input=nominal_input,
            initial_state=states_at(horizon - 1),
        )

        observer.observe(obstacle_states_at(horizon - 1))

        assert np.allclose(registry.get(metric).collisions, expected_final_collisions)
