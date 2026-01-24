from typing import Protocol, Sequence

from trajax import (
    ObstacleStateSampler,
    SampledObstacleStates,
    Sampler,
    ControlInputBatch,
    sampler,
    obstacles,
    types,
)

from numtypes import array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


class ControlInputSamplerProvider[InputSequenceT, InputBatchT](Protocol):
    def __call__(self, *, seed: int) -> Sampler[InputSequenceT, InputBatchT]:
        """Create a sampler with the given seed."""
        ...


class ObstacleStateSamplerProvider[StateT, SampleT](Protocol):
    def __call__(self, *, seed: int) -> ObstacleStateSampler[StateT, SampleT]:
        """Create a sampler with the given seed."""
        ...


class test_that_control_input_samplers_produce_same_results_when_seeded_identically:
    @staticmethod
    def cases(sampler, data, types) -> Sequence[tuple]:
        return [
            (
                provider := lambda seed: sampler.gaussian(
                    standard_deviation=array([0.5, 0.2], shape=(D_u := 2,)),
                    rollout_count=10,
                    to_batch=types.simple.control_input_batch.create,
                    seed=seed,
                ),
                input_sequence := data.control_input_sequence(
                    array(
                        [[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]], shape=(T := 3, D_u := 2)
                    )
                ),
            ),
        ]

    @mark.parametrize(
        ["provider", "input_sequence"],
        [
            *cases(sampler=sampler.numpy, data=data.numpy, types=types.numpy),
            *cases(sampler=sampler.jax, data=data.jax, types=types.jax),
        ],
    )
    def test[InputSequenceT](
        self,
        provider: ControlInputSamplerProvider[InputSequenceT, ControlInputBatch],
        input_sequence: InputSequenceT,
    ) -> None:
        sampler_1 = provider(seed=13)
        sampler_2 = provider(seed=13)
        sampler_3 = provider(seed=42)

        assert np.allclose(
            first := sampler_1.sample(around=input_sequence),
            sampler_2.sample(around=input_sequence),
        )

        assert not np.allclose(first, sampler_3.sample(around=input_sequence))

        assert np.allclose(
            second := sampler_2.sample(around=input_sequence),
            sampler_1.sample(around=input_sequence),
        )
        assert not np.allclose(second, sampler_3.sample(around=input_sequence))
        assert not np.allclose(first, second)


class test_that_sampled_control_inputs_have_correct_shape:
    @staticmethod
    def cases(sampler, data, types) -> Sequence[tuple]:
        return [
            (
                sampler.gaussian(
                    standard_deviation=array([0.3, 0.7], shape=(D_u := 2,)),
                    rollout_count=(M := 15),
                    to_batch=types.simple.control_input_batch.create,
                    seed=42,
                ),
                input_sequence := data.control_input_sequence(
                    array([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]], shape=(T := 3, D_u))
                ),
                expected_shape := (T, D_u, M),
            ),
        ]

    @mark.parametrize(
        ["sampler", "input_sequence", "expected_shape"],
        [
            *cases(sampler=sampler.numpy, data=data.numpy, types=types.numpy),
            *cases(sampler=sampler.jax, data=data.jax, types=types.jax),
        ],
    )
    def test[InputSequenceT](
        self,
        sampler: Sampler[InputSequenceT, ControlInputBatch],
        input_sequence: InputSequenceT,
        expected_shape: tuple[int, ...],
    ) -> None:
        samples = sampler.sample(around=input_sequence)

        assert samples.array.shape == expected_shape


class test_that_mean_of_sampled_control_inputs_approaches_original_when_std_is_small:
    @staticmethod
    def cases(sampler, data, types) -> Sequence[tuple]:
        return [
            (
                sampler.gaussian(
                    standard_deviation=array([1e-6, 1e-6], shape=(D_u := 2,)),
                    rollout_count=(M := 1000),
                    to_batch=types.simple.control_input_batch.create,
                    seed=42,
                ),
                input_sequence := data.control_input_sequence(
                    array([[0.0, 1.0], [1.0, 0.0], [0.5, -0.5]], shape=(T := 3, D_u))
                ),
            ),
        ]

    @mark.parametrize(
        ["sampler", "input_sequence"],
        [
            *cases(sampler=sampler.numpy, data=data.numpy, types=types.numpy),
            *cases(sampler=sampler.jax, data=data.jax, types=types.jax),
        ],
    )
    def test[InputSequenceT](
        self,
        sampler: Sampler[InputSequenceT, ControlInputBatch],
        input_sequence: InputSequenceT,
    ) -> None:
        samples = sampler.sample(around=input_sequence)

        mean_samples = np.mean(samples, axis=-1)

        assert np.allclose(mean_samples, input_sequence, atol=1e-3)


class test_that_obstacle_state_samplers_produce_same_results_when_seeded_identically:
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
            *cases(sampler=obstacles.numpy.sampler, data=data.numpy),
            *cases(sampler=obstacles.jax.sampler, data=data.jax),
        ],
    )
    def test[StateT](
        self,
        provider: ObstacleStateSamplerProvider[StateT, SampledObstacleStates],
        states: StateT,
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


class test_that_obstacle_state_sampler_returns_empty_samples_when_no_obstacles_are_present:
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
            *cases(obstacles.numpy.sampler, data.numpy),
            *cases(obstacles.jax.sampler, data.jax),
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
