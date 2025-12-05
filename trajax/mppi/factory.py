from typing import Final

from .basic import NumPyMppi, NumPyZeroPadding
from .accelerated import JaxMppi, JaxZeroPadding
from .common import NoUpdate


class mppi:
    numpy: Final = NumPyMppi.create
    jax: Final = JaxMppi.create


class update:
    class numpy:
        no_update: Final = NoUpdate

    class jax:
        no_update: Final = NoUpdate


class padding:
    class numpy:
        zero: Final = NumPyZeroPadding

    class jax:
        zero: Final = JaxZeroPadding
