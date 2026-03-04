from typing import Final

from faran.types import (
    NumPyNoiseCovariances,
    NumPyNoiseCovarianceDescription,
    JaxNoiseCovariances,
    JaxNoiseCovarianceDescription,
)
from faran.filters.kf import numpy_kalman_filter, jax_kalman_filter
from faran.filters.noise import (
    NumPyAdaptiveNoiseProvider,
    NumPyClampedNoiseProvider,
    JaxAdaptiveNoiseProvider,
    JaxClampedNoiseProvider,
    IdentityNoiseModelProvider,
)


class noise:
    """Factory namespace for noise model providers."""

    class numpy:
        adaptive: Final = NumPyAdaptiveNoiseProvider.create
        clamped: Final = NumPyClampedNoiseProvider.decorate
        identity: Final = IdentityNoiseModelProvider

        @staticmethod
        def covariances(
            *,
            process: NumPyNoiseCovarianceDescription,
            observation: NumPyNoiseCovarianceDescription,
            process_dimension: int | None = None,
            observation_dimension: int | None = None,
        ) -> NumPyNoiseCovariances:
            # TODO: Test me!
            return NumPyNoiseCovariances(
                process_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                    process, dimension=process_dimension
                ),
                observation_noise_covariance=numpy_kalman_filter.standardize_noise_covariance(
                    observation, dimension=observation_dimension
                ),
            )

    class jax:
        adaptive: Final = JaxAdaptiveNoiseProvider.create
        clamped: Final = JaxClampedNoiseProvider.decorate
        identity: Final = IdentityNoiseModelProvider

        @staticmethod
        def covariances(
            *,
            process: JaxNoiseCovarianceDescription,
            observation: JaxNoiseCovarianceDescription,
            process_dimension: int | None = None,
            observation_dimension: int | None = None,
        ) -> JaxNoiseCovariances:
            # TODO: Test me!
            return JaxNoiseCovariances(
                process_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                    process, dimension=process_dimension
                ),
                observation_noise_covariance=jax_kalman_filter.standardize_noise_covariance(
                    observation, dimension=observation_dimension
                ),
            )
