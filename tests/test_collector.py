from typing import Sequence, Callable

from trajax import (
    CollectorRegistry,
    Mppi,
    ObstacleStateObserver,
    StateSequence,
    NoCollectedDataWarning,
    collectors,
    access,
    types,
)

from numtypes import array

import numpy as np

from tests.dsl import stubs, mppi as data
from pytest import mark, warns
from pytest_subtests import SubTests


class test_that_collector_registry_discards_incomplete_data_for_time_step:
    @staticmethod
    def cases(data, types) -> Sequence[tuple]:
        states = data.state_batch(
            np.random.rand(T := 4, D_x := 5, M := 1),
        )
        obstacle_states = data.obstacle_states(
            x=(x_data := np.random.rand(T, K := 3)),
            y=(y_data := np.random.rand(T, K)),
            heading=(heading_data := np.random.rand(T, K)),
        )

        return [
            (
                registry := collectors.registry(
                    mppi := collectors.states.decorating(
                        stubs.Mppi.create(),
                        transformer=types.simple.state_sequence.of_states,
                    ),
                    # Can also be collected without a transformer
                    observer := collectors.obstacle_states.decorating(
                        stubs.ObstacleStateObserver.create()
                    ),
                ),
                mppi,
                observer,
                horizon := 4,
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                state_at := lambda t: states.at(time_step=t, rollout=0),
                obstacle_states_at := lambda t: obstacle_states.at(time_step=t),
                expected_states := lambda missing: [
                    states.at(time_step=t, rollout=0) for t in range(T - missing)
                ],
                expected_obstacle_states := lambda missing: [
                    obstacle_states.at(time_step=t) for t in range(T - missing)
                ],
            ),
        ]

    @mark.parametrize(
        [
            "registry",
            "mppi",
            "observer",
            "horizon",
            "nominal_input",
            "state_at",
            "obstacle_states_at",
            "expected_states",
            "expected_obstacle_states",
        ],
        [
            *cases(data=data.numpy, types=types.numpy),
            *cases(data=data.jax, types=types.jax),
        ],
    )
    @mark.filterwarnings("error")
    def test[StateT, InputSequenceT, ObstacleStatesForTimeStepT](
        self,
        registry: CollectorRegistry,
        mppi: Mppi[StateT, InputSequenceT],
        observer: ObstacleStateObserver[ObstacleStatesForTimeStepT],
        horizon: int,
        nominal_input: InputSequenceT,
        state_at: Callable[[int], StateT],
        obstacle_states_at: Callable[[int], ObstacleStatesForTimeStepT],
        expected_states: Callable[[int], StateSequence],
        expected_obstacle_states: Callable[[int], Sequence[ObstacleStatesForTimeStepT]],
        subtests: SubTests,
    ) -> None:
        def mppi_step(step: int) -> None:
            mppi.step(
                temperature=1.0,
                nominal_input=nominal_input,
                initial_state=state_at(step),
            )

        def observe(step: int) -> None:
            observer.observe(obstacle_states_at(step))

        for step in range(horizon - 2):
            mppi_step(step)
            observe(step)

        with subtests.test("missing obstacle state"):
            mppi_step(horizon - 2)

            assert np.allclose(registry.data(access.states), expected_states(missing=2))
            assert np.allclose(
                registry.data(access.obstacle_states),
                expected_obstacle_states(missing=2),
            )

        with subtests.test("equal number of collected states and obstacle states"):
            observe(horizon - 2)

            assert np.allclose(registry.data(access.states), expected_states(missing=1))
            assert np.allclose(
                registry.data(access.obstacle_states),
                expected_obstacle_states(missing=1),
            )

        with subtests.test("missing state"):
            observe(horizon - 1)

            assert np.allclose(registry.data(access.states), expected_states(missing=1))
            assert np.allclose(
                registry.data(access.obstacle_states),
                expected_obstacle_states(missing=1),
            )

        with subtests.test("complete data"):
            mppi_step(horizon - 1)

            assert np.allclose(registry.data(access.states), expected_states(missing=0))
            assert np.allclose(
                registry.data(access.obstacle_states),
                expected_obstacle_states(missing=0),
            )


class test_that_collector_registry_emits_warnings_when_collectors_do_not_collect_data:
    @staticmethod
    def cases(data) -> Sequence[tuple]:
        states = data.state_batch(
            np.random.rand(T := 4, D_x := 5, M := 1),
        )

        return [
            (
                registry := collectors.registry(
                    mppi := collectors.states.decorating(stubs.Mppi.create()),
                    # This collector is registered, but it never collects any data.
                    # In such cases, the registry ignores this collector and only issues a warning.
                    collectors.controls.decorating(mppi),
                ),
                mppi,
                horizon := 4,
                nominal_input := data.control_input_sequence(
                    np.random.rand(T, D_u := 2)
                ),
                state_at := lambda t: states.at(time_step=t, rollout=0),
            ),
        ]

    @mark.parametrize(
        ["registry", "mppi", "horizon", "nominal_input", "state_at"],
        [*cases(data=data.numpy), *cases(data=data.jax)],
    )
    def test[StateT, InputSequenceT, ObstacleStatesForTimeStepT](
        self,
        registry: CollectorRegistry,
        mppi: Mppi[StateT, InputSequenceT],
        horizon: int,
        nominal_input: InputSequenceT,
        state_at: Callable[[int], StateT],
    ) -> None:
        for step in range(horizon):
            mppi.step(
                temperature=1.0,
                nominal_input=nominal_input,
                initial_state=state_at(step),
            )

        with warns(NoCollectedDataWarning):
            registry.data(access.states)


class test_that_obstacle_state_collector_works_when_number_of_obstacles_varies:
    @staticmethod
    def cases(data, types) -> Sequence[tuple]:
        obstacle_states = [
            data.obstacle_states_for_time_step(
                x=array([0.0, 1.0], shape=(K_0 := 2,)),
                y=array([1.0, 2.0], shape=(K_0,)),
                heading=array([0.0, 3.0], shape=(K_0,)),
            ),
            data.obstacle_states_for_time_step(
                x=array([0.0, 1.0, 2.0], shape=(K_1 := 3,)),
                y=array([1.0, 2.0, 3.0], shape=(K_1,)),
                heading=array([0.0, 3.0, 6.0], shape=(K_1,)),
            ),
            data.obstacle_states_for_time_step(
                x=array([0.0], shape=(K_2 := 1,)),
                y=array([1.0], shape=(K_2,)),
                heading=array([0.0], shape=(K_2,)),
            ),
            data.obstacle_states_for_time_step(
                x=array([], shape=(K_3 := 0,)),
                y=array([], shape=(K_3,)),
                heading=array([], shape=(K_3,)),
            ),
        ]

        return [
            (
                registry := collectors.registry(
                    observer := collectors.obstacle_states.decorating(
                        stubs.ObstacleStateObserver.create(),
                        transformer=types.obstacle_states.of_states,
                    ),
                ),
                observer,
                horizon := (T := 4),
                obstacle_states_at := lambda t: obstacle_states[t],
                expected_obstacle_states := array(
                    [
                        [[0.0, 1.0, np.nan], [1.0, 2.0, np.nan], [0.0, 3.0, np.nan]],
                        [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.0, 3.0, 6.0]],
                        [
                            [0.0, np.nan, np.nan],
                            [1.0, np.nan, np.nan],
                            [0.0, np.nan, np.nan],
                        ],
                        [
                            [np.nan, np.nan, np.nan],
                            [np.nan, np.nan, np.nan],
                            [np.nan, np.nan, np.nan],
                        ],
                    ],
                    shape=(T, D_o := 3, K := 3),
                ),
            ),
        ]

    @mark.parametrize(
        [
            "registry",
            "observer",
            "horizon",
            "obstacle_states_at",
            "expected_obstacle_states",
        ],
        [
            *cases(data=data.numpy, types=types.numpy),
            *cases(data=data.jax, types=types.jax),
        ],
    )
    def test[ObstacleStatesForTimeStepT](
        self,
        registry: CollectorRegistry,
        observer: ObstacleStateObserver[ObstacleStatesForTimeStepT],
        horizon: int,
        obstacle_states_at: Callable[[int], ObstacleStatesForTimeStepT],
        expected_obstacle_states: Sequence[ObstacleStatesForTimeStepT],
    ) -> None:
        for step in range(horizon):
            observer.observe(obstacle_states_at(step))

        assert np.allclose(
            registry.data(access.obstacle_states),
            expected_obstacle_states,
            equal_nan=True,
        )
