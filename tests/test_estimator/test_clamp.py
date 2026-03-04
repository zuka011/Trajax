from typing import Sequence

from faran import Noise, NoiseModel, NumPyGaussianBelief, JaxGaussianBelief, noise

import numpy as np
import jax.numpy as jnp

from tests.dsl import stubs
from pytest import mark


class test_that_clamped_noise_does_not_go_below_floor:
    @staticmethod
    def cases(noise, belief, to_array) -> Sequence[tuple]:
        observation_matrix = to_array(np.eye(D_z := 3, D_x := 6))

        provider = noise.clamped(
            stubs.NoiseModelProvider.returning(
                noise.covariances(
                    process=1e-10,
                    observation=1e-10,
                    process_dimension=D_x,
                    observation_dimension=D_z,
                )
            ),
            floor=(
                floor := noise.covariances(
                    process=1e-5,
                    observation=1e-5,
                    process_dimension=D_x,
                    observation_dimension=D_z,
                )
            ),
        )

        model = provider(
            observation_matrix=observation_matrix,
            noise=noise.covariances(
                process=1.0,
                observation=1.0,
                process_dimension=D_x,
                observation_dimension=D_z,
            ),
        )

        return [
            (
                model,
                belief(mean=np.zeros(D_x), covariance=np.eye(D_x)),
                observation_matrix,
                floor,
            )
        ]

    @mark.parametrize(
        ["model", "belief", "observation", "floor"],
        [
            *cases(noise=noise.numpy, belief=NumPyGaussianBelief, to_array=np.asarray),
            *cases(noise=noise.jax, belief=JaxGaussianBelief, to_array=jnp.asarray),
        ],
    )
    def test[NoiseT: Noise, BeliefT, ObservationT](
        self,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        belief: BeliefT,
        observation: ObservationT,
        floor: NoiseT,
    ) -> None:
        result, _ = model(
            noise=floor, prediction=belief, observation=observation, state=model.state
        )

        assert np.all(result.process_noise_covariance >= floor.process_noise_covariance)
        assert np.all(
            result.observation_noise_covariance >= floor.observation_noise_covariance
        )


class test_that_clamped_noise_is_not_changed_when_noise_is_above_floor:
    @staticmethod
    def cases(noise, belief, to_array) -> Sequence[tuple]:
        observation_matrix = to_array(np.eye(D_z := 3, D_x := 6))

        provider = noise.clamped(
            stubs.NoiseModelProvider.returning(
                original := noise.covariances(
                    process=1.0,
                    observation=1.0,
                    process_dimension=D_x,
                    observation_dimension=D_z,
                )
            ),
            floor=noise.covariances(
                process=1e-5,
                observation=1e-5,
                process_dimension=D_x,
                observation_dimension=D_z,
            ),
        )

        model = provider(
            observation_matrix=observation_matrix,
            noise=noise.covariances(
                process=1.0,
                observation=1.0,
                process_dimension=D_x,
                observation_dimension=D_z,
            ),
        )

        return [
            (
                model,
                belief(mean=np.zeros(D_x), covariance=np.eye(D_x)),
                observation_matrix,
                original,
            )
        ]

    @mark.parametrize(
        ["model", "belief", "observation", "original"],
        [
            *cases(noise=noise.numpy, belief=NumPyGaussianBelief, to_array=np.asarray),
            *cases(noise=noise.jax, belief=JaxGaussianBelief, to_array=jnp.asarray),
        ],
    )
    def test[NoiseT: Noise, BeliefT, ObservationT](
        self,
        model: NoiseModel[NoiseT, BeliefT, ObservationT],
        belief: BeliefT,
        observation: ObservationT,
        original: NoiseT,
    ) -> None:
        result, _ = model(
            noise=original,
            prediction=belief,
            observation=observation,
            state=model.state,
        )

        assert np.all(
            result.process_noise_covariance == original.process_noise_covariance
        )
        assert np.all(
            result.observation_noise_covariance == original.observation_noise_covariance
        )
