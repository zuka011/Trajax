from typing import Protocol

from trajax import obstacles, ObstacleStateSampler, SampledObstacleStates

from numtypes import array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


class SamplerProvider[StateT, SampleT](Protocol):
    def __call__(self, *, seed: int) -> ObstacleStateSampler[StateT, SampleT]:
        """Create a sampler with the given seed."""
        ...


@mark.parametrize(
    ["sampler", "states"],
    [
        (
            sampler := lambda seed: obstacles.sampler.numpy.gaussian(seed=seed),
            states := data.numpy.obstacle_states(
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
        (
            sampler := lambda seed: obstacles.sampler.jax.gaussian(seed=seed),
            states := data.jax.obstacle_states(
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
    ],
)
def test_that_samplers_produce_same_results_when_seeded_identically[
    StateT,
    SampleT: SampledObstacleStates,
](sampler: SamplerProvider[StateT, SampleT], states: StateT) -> None:
    N = 5

    sampler_1 = sampler(seed=13)
    sampler_2 = sampler(seed=13)
    sampler_3 = sampler(seed=42)

    assert np.allclose(first := sampler_1(states, count=N), sampler_2(states, count=N))
    assert not np.allclose(first, sampler_3(states, count=N))

    assert np.allclose(second := sampler_2(states, count=N), sampler_1(states, count=N))
    assert not np.allclose(second, sampler_3(states, count=N))
    assert not np.allclose(first, second)


@mark.parametrize(
    ["sampler", "states", "sample_count", "expected"],
    [
        (
            sampler := obstacles.sampler.numpy.gaussian(seed=42),
            states := data.numpy.obstacle_states(
                x=np.empty(shape=(T := 4, K := 0)),
                y=np.empty(shape=(T, K)),
                heading=np.empty(shape=(T, K)),
                covariance=np.empty(shape=(T, 3, 3, K)),
            ),
            sample_count := 3,
            expected := data.numpy.obstacle_state_samples(
                x=np.empty(shape=(T, K, sample_count)),
                y=np.empty(shape=(T, K, sample_count)),
                heading=np.empty(shape=(T, K, sample_count)),
            ),
        ),
        (
            sampler := obstacles.sampler.numpy.gaussian(seed=42),
            states := data.numpy.obstacle_states(
                x=np.empty(shape=(T := 4, K := 0)),
                y=np.empty(shape=(T, K)),
                heading=np.empty(shape=(T, K)),
                # Covariance can also be completely missing.
                covariance=None,
            ),
            sample_count := 3,
            expected := data.numpy.obstacle_state_samples(
                x=np.empty(shape=(T, K, sample_count)),
                y=np.empty(shape=(T, K, sample_count)),
                heading=np.empty(shape=(T, K, sample_count)),
            ),
        ),
        (
            sampler := obstacles.sampler.jax.gaussian(seed=42),
            states := data.jax.obstacle_states(
                x=np.empty(shape=(T := 4, K := 0)),
                y=np.empty(shape=(T, K)),
                heading=np.empty(shape=(T, K)),
                covariance=np.empty(shape=(T, 3, 3, K)),
            ),
            sample_count := 3,
            expected := data.jax.obstacle_state_samples(
                x=np.empty(shape=(T, K, sample_count)),
                y=np.empty(shape=(T, K, sample_count)),
                heading=np.empty(shape=(T, K, sample_count)),
            ),
        ),
        (
            sampler := obstacles.sampler.jax.gaussian(seed=42),
            states := data.jax.obstacle_states(
                x=np.empty(shape=(T := 4, K := 0)),
                y=np.empty(shape=(T, K)),
                heading=np.empty(shape=(T, K)),
                # Covariance can also be completely missing.
                covariance=None,
            ),
            sample_count := 3,
            expected := data.jax.obstacle_state_samples(
                x=np.empty(shape=(T, K, sample_count)),
                y=np.empty(shape=(T, K, sample_count)),
                heading=np.empty(shape=(T, K, sample_count)),
            ),
        ),
    ],
)
def test_that_sampler_returns_empty_samples_when_no_obstacles_are_present[
    StateT,
    SampleT: SampledObstacleStates,
](
    sampler: ObstacleStateSampler[StateT, SampleT],
    states: StateT,
    sample_count: int,
    expected: SampleT,
) -> None:
    samples = sampler(states, count=sample_count)

    assert np.allclose(samples, expected)
