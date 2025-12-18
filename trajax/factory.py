from typing import Final

from trajax.mppi import (
    NumPyMppi,
    NumPyZeroPadding,
    JaxMppi,
    JaxZeroPadding,
    NoUpdate,
    UseOptimalControlUpdate,
)
from trajax.states import AugmentedMppi, NumPyAugmentedMppi


class mppi:
    augmented: Final = AugmentedMppi.create

    class numpy:
        base: Final = NumPyMppi.create
        augmented: Final = NumPyAugmentedMppi.create

    class jax:
        base: Final = JaxMppi.create


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
