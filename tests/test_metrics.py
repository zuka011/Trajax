from typing import Sequence, Callable, NamedTuple

from trajax import (
    MetricRegistry,
    CollisionMetric,
    ComfortMetric,
    ConstraintViolationMetric,
    TaskCompletionMetric,
    Mppi,
    ObstacleStateObserver,
    boundary,
    collectors,
    metrics,
    types,
    trajectory,
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
            assert isinstance(results.collision_detected, bool)

        with subtests.test("by name"):
            results = registry.get(metric.name)

            assert np.allclose(results.distances, expected_distances)
            assert np.allclose(results.min_distances, expected_min_distances)
            assert np.allclose(results.collisions, expected_collisions)
            assert results.collision_detected == expected_collision_detected
            assert isinstance(results.collision_detected, bool)


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
                states_at := lambda t, states=states: states.at(time_step=t, rollout=0),
                obstacle_states_at := (
                    lambda t, obstacle_states=obstacle_states: obstacle_states.at(
                        time_step=t, sample=0
                    )
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


class TaskCompletionExpectation(NamedTuple):
    completion: Sequence[bool]
    completed: bool
    completion_time: float


class test_that_task_completion_is_detected:
    @staticmethod
    def cases(data, types, trajectory) -> Sequence[tuple]:
        return [
            (
                registry := metrics.registry(
                    metric := metrics.task_completion(
                        reference=trajectory.line(
                            start=(1.0, -1.0),
                            end=(15.0, 10.0),  # Goal
                            path_length=999.0,
                        ),
                        distance_threshold=5.0,  # Within 5 meters of goal.
                        time_step_size=0.5,
                        position_extractor=lambda states: types.positions(
                            x=states.array[:, 0],
                            y=states.array[:, 1],
                        ),
                    ),
                    collectors=collectors.registry(
                        mppi := collectors.states.decorating(
                            stubs.Mppi.create(),
                            transformer=types.simple.state_sequence.of_states,
                        ),
                    ),
                ),
                metric,
                mppi,
                horizon := (T := 5),
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                states_at := lambda t, T=T: data.state_batch(
                    array(
                        [
                            [[x := 0.1], [y := 0.1], [phi := 0.0]],  # t=0
                            [[x := 5.0], [y := 5.0], [phi := 0.0]],  # t=1
                            [[x := 10.0], [y := 9.9], [phi := 0.0]],  # t=2
                            [[x := 14.0], [y := 14.0], [phi := 0.0]],  # task completed
                            [[x := 16.0], [y := 9.0], [phi := 0.0]],  # still completed
                        ],
                        shape=(T, D_x := 3, M := 1),
                    )
                ).at(time_step=t, rollout=0),
                expected := [
                    TaskCompletionExpectation(
                        completion=[False],
                        completed=False,
                        completion_time=float("inf"),
                    ),  # t=0
                    TaskCompletionExpectation(
                        completion=[False, False],
                        completed=False,
                        completion_time=float("inf"),
                    ),  # t=1
                    TaskCompletionExpectation(
                        completion=[False, False, False],
                        completed=False,
                        completion_time=float("inf"),
                    ),  # t=2
                    TaskCompletionExpectation(
                        completion=[False, False, False, True],
                        completed=True,
                        completion_time=1.5,
                    ),  # t=3 * 0.5s
                    TaskCompletionExpectation(
                        completion=[False, False, False, True, True],
                        completed=True,
                        completion_time=1.5,
                    ),  # t=4
                ],
            )
        ]

    @mark.parametrize(
        [
            "registry",
            "metric",
            "mppi",
            "horizon",
            "nominal_input",
            "states_at",
            "expected",
        ],
        [
            *cases(data=data.numpy, types=types.numpy, trajectory=trajectory.numpy),
            *cases(data=data.jax, types=types.jax, trajectory=trajectory.jax),
        ],
    )
    def test[StateT, InputSequenceT](
        self,
        registry: MetricRegistry,
        metric: TaskCompletionMetric,
        mppi: Mppi[StateT, InputSequenceT],
        horizon: int,
        nominal_input: InputSequenceT,
        states_at: Callable[[int], StateT],
        expected: Sequence[TaskCompletionExpectation],
    ) -> None:
        for step in range(horizon):
            mppi.step(
                temperature=1.0,
                nominal_input=nominal_input,
                initial_state=states_at(step),
            )

            results = registry.get(metric)

            assert results.completion.tolist() == expected[step].completion
            assert results.completed == expected[step].completed
            assert results.completion_time == expected[step].completion_time
            assert isinstance(results.completed, bool)
            assert isinstance(results.completion_time, float)


class test_that_task_efficiency_is_computed:
    @staticmethod
    def cases(data, types, trajectory) -> Sequence[tuple]:
        return [
            (
                registry := metrics.registry(
                    metric := metrics.task_completion(
                        reference=trajectory.line(
                            # Path length should not matter.
                            # Actual length and optimal distance is 5.0
                            start=(0.0, 0.0),
                            end=(3.0, 4.0),
                            path_length=999.0,
                        ),
                        distance_threshold=1.0,  # Does not matter for this test.
                        time_step_size=0.1,
                        position_extractor=lambda states: types.positions(
                            x=states.array[:, 0],
                            y=states.array[:, 1],
                        ),
                    ),
                    collectors=collectors.registry(
                        mppi := collectors.states.decorating(
                            stubs.Mppi.create(),
                            transformer=types.simple.state_sequence.of_states,
                        ),
                    ),
                ),
                metric,
                mppi,
                horizon := (T := 6),
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                states_at := lambda t, T=T: data.state_batch(
                    array(
                        [
                            [[x := 0.0], [y := 0.0], [phi := 0.0]],
                            [[x := 3.0], [y := 0.0], [phi := 0.0]],
                            [[x := 0.0], [y := 4.0], [phi := 0.0]],
                            [[x := 3.0], [y := 4.0], [phi := 0.0]],  # Goal reached
                            [[x := 3.0], [y := 4.0], [phi := 0.0]],  # Stands still
                            [[x := 5.0], [y := 4.0], [phi := 0.0]],  # Move counts
                        ],
                        shape=(T, D_x := 3, M := 1),
                    )
                ).at(time_step=t, rollout=0),
                # Efficiency is independent of whether the task was actually completed and when.
                # This just measures how the actual traveled distance compares to the optimal distance.
                expected_efficiencies := [
                    0.0,  # Super efficient!
                    0.6,  # 3.0 / 5.0
                    1.6,  # 8.0 / 5.0
                    2.2,  # 11.0 / 5.0
                    2.2,  # 11.0 / 5.0
                    2.6,  # 13.0 / 5.0
                ],
            )
        ]

    @mark.parametrize(
        [
            "registry",
            "metric",
            "mppi",
            "horizon",
            "nominal_input",
            "states_at",
            "expected_efficiencies",
        ],
        [
            *cases(data=data.numpy, types=types.numpy, trajectory=trajectory.numpy),
            *cases(data=data.jax, types=types.jax, trajectory=trajectory.jax),
        ],
    )
    def test[StateT, InputSequenceT](
        self,
        registry: MetricRegistry,
        metric: TaskCompletionMetric,
        mppi: Mppi[StateT, InputSequenceT],
        horizon: int,
        nominal_input: InputSequenceT,
        states_at: Callable[[int], StateT],
        expected_efficiencies: Sequence[float],
    ) -> None:
        for step in range(horizon):
            mppi.step(
                temperature=1.0,
                nominal_input=nominal_input,
                initial_state=states_at(step),
            )

            results = registry.get(metric)

            assert np.isclose(results.efficiency, expected_efficiencies[step])
            assert isinstance(results.efficiency, float)


class ConstraintViolationExpectation(NamedTuple):
    lateral_deviations: Sequence[float]
    boundary_distances: Sequence[float]
    violations: Sequence[bool]
    violation_detected: bool


class test_that_constraint_violation_metrics_are_computed:
    @staticmethod
    def cases(data, types, create_trajectory, create_boundary) -> Sequence[tuple]:
        reference = create_trajectory.line(
            start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0
        )

        return [
            (
                registry := metrics.registry(
                    metric := metrics.constraint_violation(
                        reference=reference,
                        boundary=create_boundary.fixed_width(
                            reference=reference,
                            position_extractor=(
                                position_extractor := lambda states: types.positions(
                                    x=states.array[:, 0],
                                    y=states.array[:, 1],
                                )
                            ),
                            left=2.0,
                            right=3.0,
                        ),
                        position_extractor=position_extractor,
                    ),
                    collectors=collectors.registry(
                        mppi := collectors.states.decorating(
                            stubs.Mppi.create(),
                            transformer=types.simple.state_sequence.of_states,
                        ),
                    ),
                ),
                metric,
                mppi,
                horizon := (T := 6),
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                states_at := lambda t, T=T: data.state_batch(
                    array(
                        [
                            [[x := 1.0], [y := 0.0], [phi := 0.0]],  # On centerline
                            [[x := 2.0], [y := 1.0], [phi := 0.0]],  # Left of center
                            [[x := 3.0], [y := -1.5], [phi := 0.0]],  # Right of center
                            [[x := 4.0], [y := -0.5], [phi := 0.0]],  # Slightly right
                            [[x := 5.0], [y := -4.0], [phi := 0.0]],  # Outside right
                            [[x := 6.0], [y := 0.5], [phi := 0.0]],  # Back inside
                        ],
                        shape=(T, D_x := 3, M := 1),
                    )
                ).at(time_step=t, rollout=0),
                # Lateral deviation: positive = right, negative = left
                # (perpendicular = [0, -1] for a horizontal line)
                # Distance to nearest boundary:
                # t=0: min(2+0, 3-0) = 2.0
                # t=1: min(2-1, 3+1) = 1.0
                # t=2: min(2+1.5, 3-1.5) = 1.5
                # t=3: min(2+0.5, 3-0.5) = 2.5
                # t=4: min(2-4.0, 3+4.0) = -2.0
                # t=5: min(2-0.5, 3+0.5) = 1.5
                expected := [
                    ConstraintViolationExpectation(
                        lateral_deviations=[0.0],
                        boundary_distances=[2.0],
                        violations=[False],
                        violation_detected=False,
                    ),
                    ConstraintViolationExpectation(
                        lateral_deviations=[0.0, -1.0],
                        boundary_distances=[2.0, 1.0],
                        violations=[False, False],
                        violation_detected=False,
                    ),
                    ConstraintViolationExpectation(
                        lateral_deviations=[0.0, -1.0, 1.5],
                        boundary_distances=[2.0, 1.0, 1.5],
                        violations=[False, False, False],
                        violation_detected=False,
                    ),
                    ConstraintViolationExpectation(
                        lateral_deviations=[0.0, -1.0, 1.5, 0.5],
                        boundary_distances=[2.0, 1.0, 1.5, 2.5],
                        violations=[False, False, False, False],
                        violation_detected=False,
                    ),
                    ConstraintViolationExpectation(
                        lateral_deviations=[0.0, -1.0, 1.5, 0.5, 4.0],
                        boundary_distances=[2.0, 1.0, 1.5, 2.5, -1.0],
                        violations=[False, False, False, False, True],
                        violation_detected=True,
                    ),
                    ConstraintViolationExpectation(
                        lateral_deviations=[0.0, -1.0, 1.5, 0.5, 4.0, -0.5],
                        boundary_distances=[2.0, 1.0, 1.5, 2.5, -1.0, 1.5],
                        violations=[False, False, False, False, True, False],
                        violation_detected=True,
                    ),
                ],
            )
        ]

    @mark.parametrize(
        [
            "registry",
            "metric",
            "mppi",
            "horizon",
            "nominal_input",
            "states_at",
            "expected",
        ],
        [
            *cases(
                data=data.numpy,
                types=types.numpy,
                create_trajectory=trajectory.numpy,
                create_boundary=boundary.numpy,
            ),
            *cases(
                data=data.jax,
                types=types.jax,
                create_trajectory=trajectory.jax,
                create_boundary=boundary.jax,
            ),
        ],
    )
    def test[StateT, InputSequenceT](
        self,
        registry: MetricRegistry,
        metric: ConstraintViolationMetric,
        mppi: Mppi[StateT, InputSequenceT],
        horizon: int,
        nominal_input: InputSequenceT,
        states_at: Callable[[int], StateT],
        expected: Sequence[ConstraintViolationExpectation],
    ) -> None:
        for step in range(horizon):
            mppi.step(
                temperature=1.0,
                nominal_input=nominal_input,
                initial_state=states_at(step),
            )

            results = registry.get(metric)

            assert np.allclose(
                results.lateral_deviations, expected[step].lateral_deviations
            )
            assert np.allclose(
                results.boundary_distances, expected[step].boundary_distances
            )
            assert np.allclose(results.violations, expected[step].violations)
            assert results.violation_detected == expected[step].violation_detected
            assert isinstance(results.violation_detected, bool)


class ComfortExpectation(NamedTuple):
    lateral_acceleration: Sequence[float]
    lateral_jerk: Sequence[float]


class test_that_comfort_metrics_are_computed:
    @staticmethod
    def cases(data, types, create_trajectory) -> Sequence[tuple]:
        reference = create_trajectory.line(
            start=(0.0, 0.0), end=(100.0, 0.0), path_length=100.0
        )

        def position_extractor(states):
            return types.positions(x=states.array[:, 0], y=states.array[:, 1])

        return [
            (  # Constant lateral deviation should yield zero acceleration and zero jerk.
                registry := metrics.registry(
                    metric := metrics.comfort(
                        reference=reference,
                        time_step_size=0.1,
                        position_extractor=position_extractor,
                    ),
                    collectors=collectors.registry(
                        mppi := collectors.states.decorating(
                            stubs.Mppi.create(),
                            transformer=types.simple.state_sequence.of_states,
                        ),
                    ),
                ),
                metric,
                mppi,
                horizon := (T := 5),
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                # Constant y = 2.0 (constant lateral offset)
                states_at := lambda t, T=T: data.state_batch(
                    array(
                        [
                            [[x := 0.0], [y := 2.0], [phi := 0.0]],
                            [[x := 10.0], [y := 2.0], [phi := 0.0]],
                            [[x := 20.0], [y := 2.0], [phi := 0.0]],
                            [[x := 30.0], [y := 2.0], [phi := 0.0]],
                            [[x := 40.0], [y := 2.0], [phi := 0.0]],
                        ],
                        shape=(T, D_x := 3, M := 1),
                    )
                ).at(time_step=t, rollout=0),
                # Constant lateral → zero acceleration → zero jerk
                expected := [
                    ComfortExpectation(
                        lateral_acceleration=[0.0],
                        lateral_jerk=[0.0],
                    ),
                    ComfortExpectation(
                        lateral_acceleration=[0.0, 0.0],
                        lateral_jerk=[0.0, 0.0],
                    ),
                    ComfortExpectation(
                        lateral_acceleration=[0.0, 0.0, 0.0],
                        lateral_jerk=[0.0, 0.0, 0.0],
                    ),
                    ComfortExpectation(
                        lateral_acceleration=[0.0, 0.0, 0.0, 0.0],
                        lateral_jerk=[0.0, 0.0, 0.0, 0.0],
                    ),
                    ComfortExpectation(
                        lateral_acceleration=[0.0, 0.0, 0.0, 0.0, 0.0],
                        lateral_jerk=[0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                ],
            ),
            (  # Linearly increasing lateral deviation should yield zero acceleration and zero jerk.
                registry := metrics.registry(
                    metric := metrics.comfort(
                        reference=reference,
                        time_step_size=0.1,
                        position_extractor=position_extractor,
                    ),
                    collectors=collectors.registry(
                        mppi := collectors.states.decorating(
                            stubs.Mppi.create(),
                            transformer=types.simple.state_sequence.of_states,
                        ),
                    ),
                ),
                metric,
                mppi,
                horizon := (T := 5),
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                # Linear y: y = t (lateral deviation increases linearly)
                # lateral = [-y] for horizontal line, so lateral = [0, 1, 2, 3, 4] * -1
                # velocity = constant -> acceleration = 0 -> jerk = 0
                states_at := lambda t, T=T: data.state_batch(
                    array(
                        [
                            [[x := 0.0], [y := 0.0], [phi := 0.0]],
                            [[x := 10.0], [y := 1.0], [phi := 0.0]],
                            [[x := 20.0], [y := 2.0], [phi := 0.0]],
                            [[x := 30.0], [y := 3.0], [phi := 0.0]],
                            [[x := 40.0], [y := 4.0], [phi := 0.0]],
                        ],
                        shape=(T, D_x := 3, M := 1),
                    )
                ).at(time_step=t, rollout=0),
                # Linear lateral → constant velocity → zero acceleration → zero jerk
                expected := [
                    ComfortExpectation(
                        lateral_acceleration=[0.0],
                        lateral_jerk=[0.0],
                    ),
                    ComfortExpectation(
                        lateral_acceleration=[0.0, 0.0],
                        lateral_jerk=[0.0, 0.0],
                    ),
                    ComfortExpectation(
                        lateral_acceleration=[0.0, 0.0, 0.0],
                        lateral_jerk=[0.0, 0.0, 0.0],
                    ),
                    ComfortExpectation(
                        lateral_acceleration=[0.0, 0.0, 0.0, 0.0],
                        lateral_jerk=[0.0, 0.0, 0.0, 0.0],
                    ),
                    ComfortExpectation(
                        lateral_acceleration=[0.0, 0.0, 0.0, 0.0, 0.0],
                        lateral_jerk=[0.0, 0.0, 0.0, 0.0, 0.0],
                    ),
                ],
            ),
        ]

    @mark.parametrize(
        [
            "registry",
            "metric",
            "mppi",
            "horizon",
            "nominal_input",
            "states_at",
            "expected",
        ],
        [
            *cases(
                data=data.numpy, types=types.numpy, create_trajectory=trajectory.numpy
            ),
            *cases(data=data.jax, types=types.jax, create_trajectory=trajectory.jax),
        ],
    )
    def test[StateT, InputSequenceT](
        self,
        registry: MetricRegistry,
        metric: ComfortMetric,
        mppi: Mppi[StateT, InputSequenceT],
        horizon: int,
        nominal_input: InputSequenceT,
        states_at: Callable[[int], StateT],
        expected: Sequence[ComfortExpectation],
    ) -> None:
        for step in range(horizon):
            mppi.step(
                temperature=1.0,
                nominal_input=nominal_input,
                initial_state=states_at(step),
            )

            results = registry.get(metric)

            assert np.allclose(
                results.lateral_acceleration,
                expected[step].lateral_acceleration,
                atol=1e-6,
            )
            assert np.allclose(
                results.lateral_jerk, expected[step].lateral_jerk, atol=1e-6
            )


class test_that_comfort_metric_measures_acceleration_when_it_is_constant:
    @staticmethod
    def cases(data, types, create_trajectory) -> Sequence[tuple]:
        reference = create_trajectory.line(
            start=(0.0, 0.0), end=(100.0, 0.0), path_length=100.0
        )

        def position_extractor(states):
            return types.positions(x=states.array[:, 0], y=states.array[:, 1])

        return [
            (  # Quadratic lateral deviation should yield constant acceleration and zero jerk.
                registry := metrics.registry(
                    metric := metrics.comfort(
                        reference=reference,
                        time_step_size=(dt := 1e-3),
                        position_extractor=position_extractor,
                    ),
                    collectors=collectors.registry(
                        mppi := collectors.states.decorating(
                            stubs.Mppi.create(),
                            transformer=types.simple.state_sequence.of_states,
                        ),
                    ),
                ),
                metric,
                mppi,
                horizon := (T := 7),
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                # Quadratic y: y = -t^2 → lateral = t^2 = [0, 1, 4, 9, 16, 25, 36]
                # velocity = 2t → acceleration = 2 (constant) → jerk = 0
                # Expected acceleration = 2 / dt^2
                states_at := lambda t, T=T: data.state_batch(
                    array(
                        [
                            [[x := 0.0], [y := 0.0], [phi := 0.0]],
                            [[x := 10.0], [y := -1.0], [phi := 0.0]],
                            [[x := 20.0], [y := -4.0], [phi := 0.0]],
                            [[x := 30.0], [y := -9.0], [phi := 0.0]],
                            [[x := 40.0], [y := -16.0], [phi := 0.0]],
                            [[x := 50.0], [y := -25.0], [phi := 0.0]],
                            [[x := 60.0], [y := -36.0], [phi := 0.0]],
                        ],
                        shape=(T, D_x := 3, M := 1),
                    )
                ).at(time_step=t, rollout=0),
                expected_acceleration := 2 / dt**2,
            ),
        ]

    @mark.parametrize(
        [
            "registry",
            "metric",
            "mppi",
            "horizon",
            "nominal_input",
            "states_at",
            "expected_acceleration",
        ],
        [
            *cases(
                data=data.numpy, types=types.numpy, create_trajectory=trajectory.numpy
            ),
            *cases(data=data.jax, types=types.jax, create_trajectory=trajectory.jax),
        ],
    )
    def test[StateT, InputSequenceT](
        self,
        registry: MetricRegistry,
        metric: ComfortMetric,
        mppi: Mppi[StateT, InputSequenceT],
        horizon: int,
        nominal_input: InputSequenceT,
        states_at: Callable[[int], StateT],
        expected_acceleration: float,
    ) -> None:
        for step in range(horizon):
            mppi.step(
                temperature=1.0,
                nominal_input=nominal_input,
                initial_state=states_at(step),
            )

        results = registry.get(metric)

        assert np.isclose(  # The middle step should be the most accurate
            results.lateral_acceleration[horizon // 2], expected_acceleration, rtol=0.1
        )


class test_that_comfort_metric_measures_jerk_when_it_is_constant:
    @staticmethod
    def cases(data, types, create_trajectory) -> Sequence[tuple]:
        reference = create_trajectory.line(
            start=(0.0, 0.0), end=(100.0, 0.0), path_length=100.0
        )

        def position_extractor(states):
            return types.positions(x=states.array[:, 0], y=states.array[:, 1])

        return [
            (  # Cubic lateral deviation should yield constant jerk.
                registry := metrics.registry(
                    metric := metrics.comfort(
                        reference=reference,
                        time_step_size=(dt := 1e-2),
                        position_extractor=position_extractor,
                    ),
                    collectors=collectors.registry(
                        mppi := collectors.states.decorating(
                            stubs.Mppi.create(),
                            transformer=types.simple.state_sequence.of_states,
                        ),
                    ),
                ),
                metric,
                mppi,
                horizon := (T := 8),
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                # Cubic y: y = t^3 → lateral = -t^3 = [0, -1, -8, -27, -64, -125, -216, -343]
                # velocity = -3t^2 → acceleration = -6t → jerk = -6 (constant)
                # Expected jerk = -6 / dt^3
                states_at := lambda t, T=T: data.state_batch(
                    array(
                        [
                            [[x := 0.0], [y := 0.0], [phi := 0.0]],
                            [[x := 10.0], [y := -1.0], [phi := 0.0]],
                            [[x := 20.0], [y := -8.0], [phi := 0.0]],
                            [[x := 30.0], [y := -27.0], [phi := 0.0]],
                            [[x := 40.0], [y := -64.0], [phi := 0.0]],
                            [[x := 50.0], [y := -125.0], [phi := 0.0]],
                            [[x := 60.0], [y := -216.0], [phi := 0.0]],
                            [[x := 70.0], [y := -343.0], [phi := 0.0]],
                        ],
                        shape=(T, D_x := 3, M := 1),
                    )
                ).at(time_step=t, rollout=0),
                expected_jerk := 6 / dt**3,
            ),
        ]

    @mark.parametrize(
        [
            "registry",
            "metric",
            "mppi",
            "horizon",
            "nominal_input",
            "states_at",
            "expected_jerk",
        ],
        [
            *cases(
                data=data.numpy, types=types.numpy, create_trajectory=trajectory.numpy
            ),
            *cases(data=data.jax, types=types.jax, create_trajectory=trajectory.jax),
        ],
    )
    def test[StateT, InputSequenceT](
        self,
        registry: MetricRegistry,
        metric: ComfortMetric,
        mppi: Mppi[StateT, InputSequenceT],
        horizon: int,
        nominal_input: InputSequenceT,
        states_at: Callable[[int], StateT],
        expected_jerk: float,
    ) -> None:
        for step in range(horizon):
            mppi.step(
                temperature=1.0,
                nominal_input=nominal_input,
                initial_state=states_at(step),
            )

        results = registry.get(metric)

        assert np.isclose(  # The middle step should be the most accurate
            results.lateral_jerk[horizon // 2], expected_jerk, rtol=0.1
        )
