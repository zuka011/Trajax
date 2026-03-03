from typing import Sequence, NamedTuple

from faran import (
    NumPyGaussianBelief,
    NumPyNoiseCovariances,
    JaxGaussianBelief,
    JaxNoiseCovariances,
    NoiseModel,
    NoiseModelProvider,
    noise,
)

from numtypes import array

import numpy as np
import jax.numpy as jnp

from tests.dsl import check
from pytest import mark


class NoiseModelInputs[BeliefT, ObservationT](NamedTuple):
    observation: ObservationT
    prediction: BeliefT


class test_that_noise_is_not_adapted_when_there_are_not_enough_observations:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(0)
        window = 5
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )

        def random_observations(count: int):
            return [rng.normal(size=(D_z, 1)) for _ in range(count)]

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=window)(
                    observation_matrix=to_array(H), noise=noise
                ),
                inputs := [
                    NoiseModelInputs(
                        observation=to_array(observation),
                        prediction=belief(
                            mean=to_array(
                                np.vstack([observation, np.zeros((D_x - D_z, 1))])
                            ),
                            covariance=to_array(np.eye(D_x)[:, :, np.newaxis] * 0.01),
                        ),
                    )
                    for observation in random_observations(count)
                ],
            )
            for count in [1, window - 1]
        ]

    @mark.parametrize(
        ["noise", "model", "inputs"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        inputs: Sequence[NoiseModelInputs[BeliefT, ObservationT]],
    ) -> None:
        state = model.state
        for observation, prediction in inputs:
            result, state = model(
                noise=noise, prediction=prediction, observation=observation, state=state
            )

        assert np.allclose(
            result.process_noise_covariance, noise.process_noise_covariance
        )
        assert np.allclose(
            result.observation_noise_covariance, noise.observation_noise_covariance
        )


class test_that_noise_is_adapted_when_there_are_enough_observations:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(0)
        window = 5
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )

        observations = [rng.normal(size=(D_z, 1)) for _ in range(window)]

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=window)(
                    observation_matrix=to_array(H), noise=noise
                ),
                inputs := [
                    NoiseModelInputs(
                        observation=to_array(observation),
                        prediction=belief(
                            mean=to_array(
                                np.vstack(
                                    [
                                        observation
                                        + rng.normal(scale=2.0, size=(D_z, 1)),
                                        np.zeros((D_x - D_z, 1)),
                                    ]
                                )
                            ),
                            covariance=to_array(np.eye(D_x)[:, :, np.newaxis] * 0.01),
                        ),
                    )
                    for observation in observations
                ],
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "inputs"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        inputs: Sequence[NoiseModelInputs[BeliefT, ObservationT]],
    ) -> None:
        state = model.state
        for observation, prediction in inputs:
            result, state = model(
                noise=noise, prediction=prediction, observation=observation, state=state
            )

        assert not np.allclose(
            result.process_noise_covariance, noise.process_noise_covariance
        )
        assert not np.allclose(
            result.observation_noise_covariance, noise.observation_noise_covariance
        )


class test_that_zero_innovation_produces_near_zero_adapted_noise:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[1.0], [2.0], [0.5], [1.0], [0.0], [0.1]], shape=(D_x, 1)
        )

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=1)(
                    observation_matrix=to_array(H), noise=noise
                ),
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis]),
                ),
                observation := to_array(H @ predicted_mean),
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "prediction", "observation"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        prediction: BeliefT,
        observation: ObservationT,
    ) -> None:
        result, _ = model(
            noise=noise,
            prediction=prediction,
            observation=observation,
            state=model.state,
        )

        R = np.asarray(result.process_noise_covariance)
        Q = np.asarray(result.observation_noise_covariance)

        # Anything less than 1e-7 is effectively zero for our purposes.
        assert np.allclose(R, 0.0, atol=1e-7), (
            "Adapted process noise covariance is not near zero for zero innovation."
        )
        assert np.allclose(Q, 0.0, atol=1e-7), (
            "Adapted observation noise covariance is not near zero for zero innovation."
        )

        # Still, it must be positive definite to avoid breaking the filter.
        assert check.is_spd(R, atol=1e-10), (
            "Adapted process noise covariance is not positive definite for zero innovation."
        )
        assert check.is_spd(Q, atol=1e-10), (
            "Adapted observation noise covariance is not positive definite for zero innovation."
        )


class test_that_adapted_process_noise_scales_quadratically_with_innovation:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[1.0], [2.0], [0.5], [1.0], [0.0], [0.1]], shape=(D_x, 1)
        )
        innovation = array([[0.5], [0.3], [0.1]], shape=(D_z, 1))

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-10),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-10),
                ),
                observation_matrix := to_array(H),
                provider := provider(window_size=1),
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis]),
                ),
                observation_1x := to_array(H @ predicted_mean + innovation),
                observation_2x := to_array(H @ predicted_mean + 2 * innovation),
            ),
        ]

    @mark.parametrize(
        [
            "noise",
            "observation_matrix",
            "provider",
            "prediction",
            "observation_1x",
            "observation_2x",
        ],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT, MatrixT](
        self,
        noise: NoiseT,
        observation_matrix: MatrixT,
        provider: NoiseModelProvider[NoiseT, BeliefT, ObservationT, MatrixT],
        prediction: BeliefT,
        observation_1x: ObservationT,
        observation_2x: ObservationT,
    ) -> None:
        model_1x = provider(observation_matrix=observation_matrix, noise=noise)
        model_2x = provider(observation_matrix=observation_matrix, noise=noise)

        result_1x, _ = model_1x(
            noise=noise,
            prediction=prediction,
            observation=observation_1x,
            state=model_1x.state,
        )
        result_2x, _ = model_2x(
            noise=noise,
            prediction=prediction,
            observation=observation_2x,
            state=model_2x.state,
        )

        assert np.allclose(
            np.asarray(result_2x.process_noise_covariance),
            4.0 * np.asarray(result_1x.process_noise_covariance),
        )


class test_that_all_innovation_is_attributed_to_observation_noise_when_state_is_certain:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[1.0], [2.0], [0.5], [1.0], [0.0], [0.1]], shape=(D_x, 1)
        )
        innovation = array([[0.5], [0.3], [0.1]], shape=(D_z, 1))

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=1)(
                    observation_matrix=to_array(H), noise=noise
                ),
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis] * 1e-12),
                ),
                observation := to_array(H @ predicted_mean + innovation),
                expected_observation_noise := to_array(innovation @ innovation.T),
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "prediction", "observation", "expected_observation_noise"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        prediction: BeliefT,
        observation: ObservationT,
        expected_observation_noise: ObservationT,
    ) -> None:
        result, _ = model(
            noise=noise,
            prediction=prediction,
            observation=observation,
            state=model.state,
        )

        assert np.allclose(
            np.asarray(result.observation_noise_covariance),
            np.asarray(expected_observation_noise),
            atol=1e-6,
        )
        assert np.allclose(
            np.asarray(result.process_noise_covariance),
            0.0,
            atol=1e-6,
        )


class test_that_repeated_identical_innovations_match_single_innovation:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[1.0], [2.0], [0.5], [1.0], [0.0], [0.1]], shape=(D_x, 1)
        )
        innovation = array([[0.5], [0.3], [0.1]], shape=(D_z, 1))
        observation = H @ predicted_mean + innovation

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x)),
                    observation_noise_covariance=to_array(np.eye(D_z)),
                ),
                observation_matrix := to_array(H),
                provider := provider,
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis]),
                ),
                observation := to_array(observation),
            ),
        ]

    @mark.parametrize(
        ["noise", "observation_matrix", "provider", "prediction", "observation"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT, MatrixT](
        self,
        noise: NoiseT,
        observation_matrix: MatrixT,
        provider: NoiseModelProvider[NoiseT, BeliefT, ObservationT, MatrixT],
        prediction: BeliefT,
        observation: ObservationT,
    ) -> None:
        single = provider(window_size=1)(
            observation_matrix=observation_matrix, noise=noise
        )
        result_single, _ = single(
            noise=noise,
            prediction=prediction,
            observation=observation,
            state=single.state,
        )

        repeated = provider(window_size=5)(
            observation_matrix=observation_matrix, noise=noise
        )
        state = repeated.state
        for _ in range(5):
            result_repeated, state = repeated(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state,
            )

        assert np.allclose(
            np.asarray(result_repeated.process_noise_covariance),
            np.asarray(result_single.process_noise_covariance),
        )
        assert np.allclose(
            np.asarray(result_repeated.observation_noise_covariance),
            np.asarray(result_single.observation_noise_covariance),
        )


class test_that_orthogonal_unit_innovations_produce_isotropic_adapted_process_noise:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        predicted_mean = array(
            [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], shape=(D_x, 1)
        )

        def unit_observation(*, dimensions: int, index: int):
            return to_array(np.eye(dimensions)[:, index : index + 1])

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-12),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-12),
                ),
                model := provider(window_size=D_z)(
                    observation_matrix=to_array(H), noise=noise
                ),
                prediction := belief(
                    mean=to_array(predicted_mean),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis]),
                ),
                observations := [
                    unit_observation(dimensions=D_z, index=i) for i in range(D_z)
                ],
                observed_dimensions := D_z,
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "prediction", "observations", "observed_dimensions"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        prediction: BeliefT,
        observations: Sequence[ObservationT],
        observed_dimensions: int,
    ) -> None:
        state = model.state
        for observation in observations:
            result, state = model(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state,
            )

        R = np.asarray(result.process_noise_covariance)
        observed_diagonal = np.diag(R)[:observed_dimensions]

        # Isotropic innovations → equal adapted noise for all observed states.
        assert np.allclose(observed_diagonal[0], observed_diagonal[1])
        assert np.allclose(observed_diagonal[0], observed_diagonal[2])

        # Observed states should have picked up some process noise.
        assert observed_diagonal[0] > 0

        # Unobserved states should have zero adapted process noise
        # (no information flows to them from isotropic observed innovations).
        unobserved_diagonal = np.diag(R)[observed_dimensions:]
        assert np.allclose(unobserved_diagonal, 0.0, atol=1e-10)


class test_that_two_models_from_same_provider_have_independent_state:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(0)
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        window = 5

        observations = [rng.normal(size=(D_z, 1)) for _ in range(window + 1)]

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                observation_matrix := to_array(H),
                provider := provider(window_size=window),
                inputs := [
                    NoiseModelInputs(
                        observation=to_array(observation),
                        prediction=belief(
                            mean=to_array(
                                np.vstack([observation, np.zeros((D_x - D_z, 1))])
                            ),
                            covariance=to_array(np.eye(D_x)[:, :, np.newaxis] * 0.01),
                        ),
                    )
                    for observation in observations
                ],
            ),
        ]

    @mark.parametrize(
        ["noise", "observation_matrix", "provider", "inputs"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT, MatrixT](
        self,
        noise: NoiseT,
        observation_matrix: MatrixT,
        provider: NoiseModelProvider[NoiseT, BeliefT, ObservationT, MatrixT],
        inputs: Sequence[NoiseModelInputs[ObservationT, BeliefT]],
    ) -> None:
        model_a = provider(observation_matrix=observation_matrix, noise=noise)
        model_b = provider(observation_matrix=observation_matrix, noise=noise)

        state_a = model_a.state
        for observation, prediction in inputs[:-1]:
            _, state_a = model_a(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state_a,
            )

        # This one received less than `window` observations.
        observation, prediction = inputs[-1]
        state_b = model_b.state
        result, _ = model_b(
            noise=noise, prediction=prediction, observation=observation, state=state_b
        )

        # So we expect the noise to be unchanged.
        assert np.allclose(
            result.process_noise_covariance, noise.process_noise_covariance
        )
        assert np.allclose(
            result.observation_noise_covariance, noise.observation_noise_covariance
        )


class test_that_adapted_noise_is_spd:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(0)
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )
        window = 5

        observations = [rng.normal(size=(D_z, 1)) for _ in range(window)]

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=window)(
                    observation_matrix=to_array(H), noise=noise
                ),
                inputs := [
                    NoiseModelInputs(
                        observation=to_array(observation),
                        prediction=belief(
                            mean=to_array(
                                np.vstack(
                                    [
                                        observation
                                        + rng.normal(scale=0.1, size=(D_z, 1)),
                                        np.zeros((D_x - D_z, 1)),
                                    ]
                                )
                            ),
                            covariance=to_array(np.eye(D_x)[:, :, np.newaxis] * 0.01),
                        ),
                    )
                    for observation in observations
                ],
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "inputs"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        inputs: Sequence[NoiseModelInputs[ObservationT, BeliefT]],
    ) -> None:
        state = model.state
        for observation, prediction in inputs:
            result, state = model(
                noise=noise, prediction=prediction, observation=observation, state=state
            )

        R = np.asarray(result.process_noise_covariance)
        Q = np.asarray(result.observation_noise_covariance)

        assert check.is_spd(R, atol=1e-10), (
            "Adapted process noise covariance is not positive definite"
        )
        assert check.is_spd(Q, atol=1e-10), (
            "Adapted observation noise covariance is not positive definite"
        )


class test_that_nan_prediction_returns_noise_unchanged:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        D_x, D_z, window = 6, 3, 5
        H = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
            dtype=np.float64,
        )

        return [
            (
                noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x) * 1e-8),
                    observation_noise_covariance=to_array(np.eye(D_z) * 1e-8),
                ),
                model := provider(window_size=window)(
                    observation_matrix=to_array(H), noise=noise
                ),
                nan_prediction := belief(
                    mean=to_array(np.full((D_x, 1), np.nan)),
                    covariance=to_array(np.full((D_x, D_x, 1), np.nan)),
                ),
                observation := to_array(np.zeros((D_z, 1))),
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "nan_prediction", "observation"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        nan_prediction: BeliefT,
        observation: ObservationT,
    ) -> None:
        result, _ = model(
            noise=noise,
            prediction=nan_prediction,
            observation=observation,
            state=model.state,
        )

        assert np.allclose(
            result.process_noise_covariance, noise.process_noise_covariance
        )
        assert np.allclose(
            result.observation_noise_covariance, noise.observation_noise_covariance
        )


class test_that_observations_older_than_window_size_do_not_affect_noise:
    @staticmethod
    def cases(provider, covariances, belief, to_array) -> Sequence[tuple]:
        rng = np.random.default_rng(42)
        window = 3
        H = array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            shape=(D_z := 3, D_x := 6),
        )

        shared_observations = [rng.normal(size=(D_z, 1)) for _ in range(window)]
        old_observation_a = rng.normal(size=(D_z, 1)) * 100.0
        old_observation_b = rng.normal(size=(D_z, 1)) * 0.001

        def input_from(observation):
            return NoiseModelInputs(
                observation=to_array(observation),
                prediction=belief(
                    mean=to_array(np.zeros((D_x, 1))),
                    covariance=to_array(np.eye(D_x)[:, :, np.newaxis] * 0.01),
                ),
            )

        return [
            (
                initial_noise := covariances(
                    process_noise_covariance=to_array(np.eye(D_x)),
                    observation_noise_covariance=to_array(np.eye(D_z)),
                ),
                model := provider(window_size=window)(
                    observation_matrix=to_array(H), noise=initial_noise
                ),
                inputs_a := [
                    input_from(old_observation_a),
                    *[input_from(it) for it in shared_observations],
                ],
                inputs_b := [
                    input_from(old_observation_b),
                    *[input_from(it) for it in shared_observations],
                ],
            ),
        ]

    @mark.parametrize(
        ["noise", "model", "inputs_a", "inputs_b"],
        [
            *cases(
                provider=noise.numpy.adaptive,
                covariances=NumPyNoiseCovariances,
                belief=NumPyGaussianBelief,
                to_array=np.asarray,
            ),
            *cases(
                provider=noise.jax.adaptive,
                covariances=JaxNoiseCovariances,
                belief=JaxGaussianBelief,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test[NoiseT, BeliefT, ObservationT](
        self,
        noise: NoiseT,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        inputs_a: Sequence[NoiseModelInputs[BeliefT, ObservationT]],
        inputs_b: Sequence[NoiseModelInputs[BeliefT, ObservationT]],
    ) -> None:
        state_a = model.state
        for observation, prediction in inputs_a:
            result_a, state_a = model(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state_a,
            )

        state_b = model.state
        for observation, prediction in inputs_b:
            result_b, state_b = model(
                noise=noise,
                prediction=prediction,
                observation=observation,
                state=state_b,
            )

        assert np.allclose(
            result_a.process_noise_covariance,
            result_b.process_noise_covariance,
            atol=1e-6,
        )
        assert np.allclose(
            result_a.observation_noise_covariance,
            result_b.observation_noise_covariance,
            atol=1e-6,
        )
