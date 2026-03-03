from typing import Final

from faran.filters.noise.basic import (
    NumPyAdaptiveNoiseProvider,
    NumPyClampedNoiseProvider,
)
from faran.filters.noise.accelerated import (
    JaxAdaptiveNoiseProvider,
    JaxClampedNoiseProvider,
)
from faran.filters.noise.common import IdentityNoiseModelProvider


class noise:
    """Factory namespace for noise model providers."""

    class numpy:
        adaptive: Final = NumPyAdaptiveNoiseProvider.create
        clamped: Final = NumPyClampedNoiseProvider.decorate
        identity: Final = IdentityNoiseModelProvider

    class jax:
        adaptive: Final = JaxAdaptiveNoiseProvider.create
        clamped: Final = JaxClampedNoiseProvider.decorate
        identity: Final = IdentityNoiseModelProvider
