from typing import Protocol, Sequence

from trajax import obstacles, ObstacleStateSampler, SampledObstacleStates

from numtypes import array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


class SamplerProvider[StateT, SampleT](Protocol):
    def __call__(self, *, seed: int) -> ObstacleStateSampler[StateT, SampleT]:
        """Create a sampler with the given seed."""
        ...


class test_that_samplers_produce_same_results_when_seeded_identically:
    @staticmethod
    def cases(sampler, data) -> Sequence[tuple]:
        return [
            (
                provider := lambda seed: sampler.gaussian(seed=seed),
                states := data.obstacle_states(
                    x=array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], shape=(T := 2, K := 3)),
                    y=array([[4.0, 2.0, 3.0], [6.0, 4.0, 5.0]], shape=(T, K)),
                    heading=array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]], shape=(T, K)),
                    covariance=array(
                        [
                            [
                                [[1.0, 1.2, 0.8], [0.2, 0.3, 0.1], [0.1, 0.2, 0.1]],
                                [[0.2, 0.3, 0.1], [1.0, 1.4, 0.9], [0.1, 0.2, 0.1]],
                                [[0.1, 0.2, 0.1], [0.1, 0.2, 0.1], [1.0, 1.3, 0.7]],
                            ],
                            [
                                [[1.2, 1.4, 1.0], [0.3, 0.4, 0.2], [0.2, 0.3, 0.2]],
                                [[0.3, 0.4, 0.2], [1.3, 1.6, 1.1], [0.2, 0.3, 0.2]],
                                [[0.2, 0.3, 0.2], [0.2, 0.3, 0.2], [1.2, 1.5, 0.9]],
                            ],
                        ],
                        shape=(T, 3, 3, K),
                    ),
                ),
            ),
        ]

    @mark.parametrize(
        ["provider", "states"],
        [
            *cases(obstacles.sampler.numpy, data.numpy),
            *cases(obstacles.sampler.jax, data.jax),
        ],
    )
    def test[StateT, SampleT: SampledObstacleStates](
        self, provider: SamplerProvider[StateT, SampleT], states: StateT
    ) -> None:
        N = 5

        sampler_1 = provider(seed=13)
        sampler_2 = provider(seed=13)
        sampler_3 = provider(seed=42)

        assert np.allclose(
            first := sampler_1(states, count=N), sampler_2(states, count=N)
        )
        assert not np.allclose(first, sampler_3(states, count=N))

        assert np.allclose(
            second := sampler_2(states, count=N), sampler_1(states, count=N)
        )
        assert not np.allclose(second, sampler_3(states, count=N))
        assert not np.allclose(first, second)


class test_that_sampler_returns_empty_samples_when_no_obstacles_are_present:
    @staticmethod
    def cases(sampler, data) -> Sequence[tuple]:
        return [
            (
                sampler.gaussian(seed=42),
                states := data.obstacle_states(
                    x=np.empty(shape=(T := 4, K := 0)),
                    y=np.empty(shape=(T, K)),
                    heading=np.empty(shape=(T, K)),
                    covariance=np.empty(shape=(T, 3, 3, K)),
                ),
                sample_count := 3,
                expected := data.obstacle_state_samples(
                    x=np.empty(shape=(T, K, sample_count)),
                    y=np.empty(shape=(T, K, sample_count)),
                    heading=np.empty(shape=(T, K, sample_count)),
                ),
            ),
            (
                sampler.gaussian(seed=42),
                states := data.obstacle_states(
                    x=np.empty(shape=(T := 4, K := 0)),
                    y=np.empty(shape=(T, K)),
                    heading=np.empty(shape=(T, K)),
                    # Covariance can also be completely missing.
                    covariance=None,
                ),
                sample_count := 3,
                expected := data.obstacle_state_samples(
                    x=np.empty(shape=(T, K, sample_count)),
                    y=np.empty(shape=(T, K, sample_count)),
                    heading=np.empty(shape=(T, K, sample_count)),
                ),
            ),
        ]

    @mark.parametrize(
        ["sampler", "states", "sample_count", "expected"],
        [
            *cases(obstacles.sampler.numpy, data.numpy),
            *cases(obstacles.sampler.jax, data.jax),
        ],
    )
    def test[StateT, SampleT: SampledObstacleStates](
        self,
        sampler: ObstacleStateSampler[StateT, SampleT],
        states: StateT,
        sample_count: int,
        expected: SampleT,
    ) -> None:
        samples = sampler(states, count=sample_count)

        assert np.allclose(samples, expected)
