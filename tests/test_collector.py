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
                    states=(
                        mppi := collectors.states.decorating(
                            stubs.Mppi.create(),
                            transformer=types.simple.state_sequence.of_states,
                        )
                    ),
                    # The keyword argument name does not matter here.
                    obstacles=(  # Can also be collected without a transformer
                        observer := collectors.obstacle_states.decorating(
                            stubs.ObstacleStateObserver.create()
                        )
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
                    states=(mppi := collectors.states.decorating(stubs.Mppi.create())),
                    # This collector is registered, but it never collects any data.
                    # In such cases, the registry ignores this collector and only issues a warning.
                    ignored=collectors.controls.decorating(mppi),
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
