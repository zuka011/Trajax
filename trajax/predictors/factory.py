from typing import Final

from trajax.predictors.common import (
    StaticPredictor,
    CurvilinearPredictor,
    CovarianceResizing,
)
from trajax.predictors.propagators import (
    NumPyLinearCovariancePropagator,
    JaxLinearCovariancePropagator,
    NumPyEkfCovariancePropagator,
    JaxEkfCovariancePropagator,
)
from trajax.predictors.covariance import (
    NumPyConstantVarianceProvider,
    NumPyConstantCovarianceProvider,
    NumPyFullStateVarianceProvider,
    NumPyZeroProcessNoiseProvider,
    NumPyFullStateZeroProcessNoiseProvider,
    NumPyCovarianceProviderComposite,
    NumPyIsotropicCovarianceProvider,
    JaxConstantVarianceProvider,
    JaxConstantCovarianceProvider,
    JaxFullStateVarianceProvider,
    JaxZeroProcessNoiseProvider,
    JaxFullStateZeroProcessNoiseProvider,
    JaxCovarianceProviderComposite,
    JaxIsotropicCovarianceProvider,
)


class predictor:
    """Factory namespace for creating obstacle motion predictors."""

    class numpy:
        curvilinear: Final = CurvilinearPredictor.create
        static: Final = StaticPredictor.create

    class jax:
        curvilinear: Final = CurvilinearPredictor.create
        static: Final = StaticPredictor.create


class propagator:
    """Factory namespace for creating covariance propagators and providers."""

    class numpy:
        linear: Final = NumPyLinearCovariancePropagator.create
        ekf: Final = NumPyEkfCovariancePropagator.create

        class covariance:
            # Unified covariance providers (preferred)
            composite: Final = NumPyCovarianceProviderComposite.create
            isotropic: Final = NumPyIsotropicCovarianceProvider.create

            # Individual covariance providers (for custom composition)
            constant_variance: Final = NumPyConstantVarianceProvider.create
            constant_covariance: Final = NumPyConstantCovarianceProvider.create
            full_state_variance: Final = NumPyFullStateVarianceProvider.create
            zero_process_noise: Final = NumPyZeroProcessNoiseProvider.create
            full_state_zero_process_noise: Final = (
                NumPyFullStateZeroProcessNoiseProvider.create
            )

            resize: Final = CovarianceResizing.create

    class jax:
        linear: Final = JaxLinearCovariancePropagator.create
        ekf: Final = JaxEkfCovariancePropagator.create

        class covariance:
            # TODO: Review!
            # Unified covariance providers (preferred)
            composite: Final = JaxCovarianceProviderComposite.create
            isotropic: Final = JaxIsotropicCovarianceProvider.create

            # TODO: Review!
            # Individual covariance providers (for custom composition)
            constant_variance: Final = JaxConstantVarianceProvider.create
            constant_covariance: Final = JaxConstantCovarianceProvider.create
            full_state_variance: Final = JaxFullStateVarianceProvider.create
            zero_process_noise: Final = JaxZeroProcessNoiseProvider.create
            full_state_zero_process_noise: Final = (
                JaxFullStateZeroProcessNoiseProvider.create
            )

            resize: Final = CovarianceResizing.create
