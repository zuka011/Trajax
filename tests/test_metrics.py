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
                        states=(
                            mppi := collectors.states.decorating(stubs.Mppi.create()),
                            types.simple.state_sequence.of_states,
                        ),
                        obstacles=(
                            observer := collectors.obstacles.decorating(
                                stubs.ObstacleStateObserver.create()
                            ),
                            types.obstacle_states.of_states,
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
