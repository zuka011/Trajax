from typing import Final

from trajax.mppi.basic import NumPyMppi, ZeroPadding as NumPyZeroPadding
from trajax.mppi.accelerated import JaxMppi, ZeroPadding as JaxZeroPadding
from trajax.mppi.common import NoUpdate, UseOptimalControlUpdate


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
        zero: Final = NumPyZeroPadding

    class jax:
        zero: Final = JaxZeroPadding
