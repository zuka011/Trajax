from typing import Final

from .basic import NumPyMppi
from .accelerated import JaxMppi
from .common import NoUpdate, UseOptimalControlUpdate


class mppi:
    numpy: Final = NumPyMppi.create
    jax: Final = JaxMppi.create


class update:
    class numpy:
        no_update: Final = NoUpdate
        use_optimal_control: Final = UseOptimalControlUpdate

    class jax:
        no_update: Final = NoUpdate
        use_optimal_control: Final = UseOptimalControlUpdate


class padding:
    class numpy:
        zero: Final = NumPyMppi.ZeroPadding

    class jax:
        zero: Final = JaxMppi.ZeroPadding
