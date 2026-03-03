from faran import (
    NumPyGaussianBelief,
    NumPyNoiseCovariances,
    JaxGaussianBelief,
    JaxNoiseCovariances,
    noise,
)


import numpy as np
import jax.numpy as jnp

from pytest import mark


def make_prediction(to_array, *, state_dimension: int):
    return {
        "prediction": NumPyGaussianBelief(
            mean=to_array(np.zeros((state_dimension, 1))),
            covariance=to_array(np.eye(state_dimension)[:, :, np.newaxis]),
        )
        if to_array is np.asarray
        else JaxGaussianBelief(
            mean=to_array(np.zeros((state_dimension, 1))),
            covariance=to_array(np.eye(state_dimension)[:, :, np.newaxis]),
        ),
        "observation": to_array(np.zeros((3, 1))),
    }


class StubNoiseModel:
    """A noise model that always returns fixed covariances."""

    def __init__(self, fixed_noise):
        self._fixed_noise = fixed_noise

    @property
    def state(self) -> None:
        return None

    def __call__(self, *, noise, prediction, observation, state=None):
        return self._fixed_noise, state


class StubNoiseModelProvider:
    """A provider that always returns a StubNoiseModel with the given noise."""

    def __init__(self, fixed_noise):
        self._fixed_noise = fixed_noise

    def __call__(self, *, observation_matrix, noise):
        return StubNoiseModel(self._fixed_noise)


class test_that_clamped_noise_clamps_diagonals_below_floor:
    @staticmethod
    def cases(clamped_factory, covariances, to_array):
        D_x = 6
        D_z = 3
        observation_matrix = to_array(np.eye(D_z, D_x))
        floor = covariances(
            process_noise_covariance=to_array(np.diag([1e-5] * D_x)),
            observation_noise_covariance=to_array(np.diag([1e-5] * D_z)),
        )
        below_floor = covariances(
            process_noise_covariance=to_array(np.diag([1e-10] * D_x)),
            observation_noise_covariance=to_array(np.diag([1e-10] * D_z)),
        )

        inner_provider = StubNoiseModelProvider(below_floor)
        initial_noise = covariances(
            process_noise_covariance=to_array(np.eye(D_x)),
            observation_noise_covariance=to_array(np.eye(D_z)),
        )

        clamped_provider = clamped_factory(floor=floor, inner=inner_provider)
        clamped_model = clamped_provider(
            observation_matrix=observation_matrix, noise=initial_noise
        )

        return [(clamped_model, floor, D_x, D_z, to_array)]

    @mark.parametrize(
        ["clamped_model", "floor", "D_x", "D_z", "to_array"],
        [
            *cases(
                clamped_factory=noise.numpy.clamped,
                covariances=NumPyNoiseCovariances,
                to_array=np.asarray,
            ),
            *cases(
                clamped_factory=noise.jax.clamped,
                covariances=JaxNoiseCovariances,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test(self, clamped_model, floor, D_x, D_z, to_array) -> None:
        result, _state = clamped_model(
            **make_prediction(to_array, state_dimension=D_x),
            noise=NumPyNoiseCovariances(
                process_noise_covariance=np.eye(D_x),
                observation_noise_covariance=np.eye(D_z),
            )
            if to_array is np.asarray
            else JaxNoiseCovariances(
                process_noise_covariance=jnp.eye(D_x),
                observation_noise_covariance=jnp.eye(D_z),
            ),
            state=clamped_model.state,
        )

        process_diag = np.asarray(np.diag(result.process_noise_covariance))
        observation_diag = np.asarray(np.diag(result.observation_noise_covariance))
        floor_process_diag = np.asarray(np.diag(floor.process_noise_covariance))
        floor_observation_diag = np.asarray(np.diag(floor.observation_noise_covariance))

        assert np.all(process_diag >= floor_process_diag - 1e-15)
        assert np.all(observation_diag >= floor_observation_diag - 1e-15)
        np.testing.assert_allclose(process_diag, floor_process_diag)
        np.testing.assert_allclose(observation_diag, floor_observation_diag)


class test_that_clamped_noise_passes_through_values_above_floor:
    @staticmethod
    def cases(clamped_factory, covariances, to_array):
        D_x = 6
        D_z = 3
        observation_matrix = to_array(np.eye(D_z, D_x))
        floor = covariances(
            process_noise_covariance=to_array(np.diag([1e-5] * D_x)),
            observation_noise_covariance=to_array(np.diag([1e-5] * D_z)),
        )
        above_floor = covariances(
            process_noise_covariance=to_array(np.diag([1.0] * D_x)),
            observation_noise_covariance=to_array(np.diag([1.0] * D_z)),
        )

        inner_provider = StubNoiseModelProvider(above_floor)
        initial_noise = covariances(
            process_noise_covariance=to_array(np.eye(D_x)),
            observation_noise_covariance=to_array(np.eye(D_z)),
        )

        clamped_provider = clamped_factory(floor=floor, inner=inner_provider)
        clamped_model = clamped_provider(
            observation_matrix=observation_matrix, noise=initial_noise
        )

        return [(clamped_model, above_floor, D_x, D_z, to_array)]

    @mark.parametrize(
        ["clamped_model", "above_floor", "D_x", "D_z", "to_array"],
        [
            *cases(
                clamped_factory=noise.numpy.clamped,
                covariances=NumPyNoiseCovariances,
                to_array=np.asarray,
            ),
            *cases(
                clamped_factory=noise.jax.clamped,
                covariances=JaxNoiseCovariances,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test(self, clamped_model, above_floor, D_x, D_z, to_array) -> None:
        result, _state = clamped_model(
            **make_prediction(to_array, state_dimension=D_x),
            noise=NumPyNoiseCovariances(
                process_noise_covariance=np.eye(D_x),
                observation_noise_covariance=np.eye(D_z),
            )
            if to_array is np.asarray
            else JaxNoiseCovariances(
                process_noise_covariance=jnp.eye(D_x),
                observation_noise_covariance=jnp.eye(D_z),
            ),
            state=clamped_model.state,
        )

        np.testing.assert_allclose(
            result.process_noise_covariance,
            above_floor.process_noise_covariance,
        )
        np.testing.assert_allclose(
            result.observation_noise_covariance,
            above_floor.observation_noise_covariance,
        )


class test_that_clamped_noise_delegates_to_inner_model:
    @staticmethod
    def cases(clamped_factory, covariances, to_array):
        D_x = 6
        D_z = 3
        observation_matrix = to_array(np.eye(D_z, D_x))
        inner_result = covariances(
            process_noise_covariance=to_array(np.diag([0.5, 0.1, 0.2, 0.3, 0.4, 0.6])),
            observation_noise_covariance=to_array(np.diag([0.7, 0.8, 0.9])),
        )
        floor = covariances(
            process_noise_covariance=to_array(np.diag([1e-10] * D_x)),
            observation_noise_covariance=to_array(np.diag([1e-10] * D_z)),
        )

        inner_provider = StubNoiseModelProvider(inner_result)
        initial_noise = covariances(
            process_noise_covariance=to_array(np.eye(D_x)),
            observation_noise_covariance=to_array(np.eye(D_z)),
        )

        clamped_provider = clamped_factory(floor=floor, inner=inner_provider)
        clamped_model = clamped_provider(
            observation_matrix=observation_matrix, noise=initial_noise
        )

        return [(clamped_model, inner_result, D_x, to_array)]

    @mark.parametrize(
        ["clamped_model", "inner_result", "D_x", "to_array"],
        [
            *cases(
                clamped_factory=noise.numpy.clamped,
                covariances=NumPyNoiseCovariances,
                to_array=np.asarray,
            ),
            *cases(
                clamped_factory=noise.jax.clamped,
                covariances=JaxNoiseCovariances,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test(self, clamped_model, inner_result, D_x, to_array) -> None:
        result, _state = clamped_model(
            **make_prediction(to_array, state_dimension=D_x),
            noise=NumPyNoiseCovariances(
                process_noise_covariance=np.eye(D_x),
                observation_noise_covariance=np.eye(3),
            )
            if to_array is np.asarray
            else JaxNoiseCovariances(
                process_noise_covariance=jnp.eye(D_x),
                observation_noise_covariance=jnp.eye(3),
            ),
            state=clamped_model.state,
        )

        np.testing.assert_allclose(
            result.process_noise_covariance,
            inner_result.process_noise_covariance,
        )
        np.testing.assert_allclose(
            result.observation_noise_covariance,
            inner_result.observation_noise_covariance,
        )


class test_that_clamped_noise_wrapping_identity_returns_floor_when_noise_is_below:
    @staticmethod
    def cases(clamped_factory, identity_provider, covariances, to_array):
        D_x = 6
        D_z = 3
        observation_matrix = to_array(np.eye(D_z, D_x))
        floor = covariances(
            process_noise_covariance=to_array(
                np.diag([1e-8, 1e-8, 1e-8, 1e-5, 1e-5, 1e-5])
            ),
            observation_noise_covariance=to_array(np.diag([1e-8] * D_z)),
        )
        below_floor_noise = covariances(
            process_noise_covariance=to_array(np.diag([1e-10] * D_x)),
            observation_noise_covariance=to_array(np.diag([1e-10] * D_z)),
        )

        clamped_provider = clamped_factory(floor=floor, inner=identity_provider())
        clamped_model = clamped_provider(
            observation_matrix=observation_matrix, noise=below_floor_noise
        )

        return [(clamped_model, floor, below_floor_noise, D_x, to_array)]

    @mark.parametrize(
        ["clamped_model", "floor", "below_floor_noise", "D_x", "to_array"],
        [
            *cases(
                clamped_factory=noise.numpy.clamped,
                identity_provider=noise.numpy.identity,
                covariances=NumPyNoiseCovariances,
                to_array=np.asarray,
            ),
            *cases(
                clamped_factory=noise.jax.clamped,
                identity_provider=noise.jax.identity,
                covariances=JaxNoiseCovariances,
                to_array=jnp.asarray,
            ),
        ],
    )
    def test(self, clamped_model, floor, below_floor_noise, D_x, to_array) -> None:
        result, _state = clamped_model(
            **make_prediction(to_array, state_dimension=D_x),
            noise=below_floor_noise,
            state=clamped_model.state,
        )

        process_diag = np.asarray(np.diag(result.process_noise_covariance))
        floor_process_diag = np.asarray(np.diag(floor.process_noise_covariance))

        np.testing.assert_allclose(process_diag, floor_process_diag)
