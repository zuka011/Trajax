from typing import Final

from trajax.mppi import (
    NumPyMppi,
    NumPyZeroPadding,
    JaxMppi,
    JaxZeroPadding,
    NoUpdate,
    UseOptimalControlUpdate,
    NoFilter,
    NumPySavGolFilter,
    JaxSavGolFilter,
)
from trajax.states import NumPyAugmentedMppi, JaxAugmentedMppi
from trajax.mpcc import NumPyMpccMppi, JaxMpccMppi


class mppi:
    class numpy:
        base: Final = NumPyMppi.create
        augmented: Final = NumPyAugmentedMppi.create
        mpcc: Final = NumPyMpccMppi.create

    class jax:
        base: Final = JaxMppi.create
        augmented: Final = JaxAugmentedMppi.create
        mpcc: Final = JaxMpccMppi.create


class update:
    class numpy:
        none: Final = NoUpdate
        use_optimal_control: Final = UseOptimalControlUpdate

    class jax:
        none: Final = NoUpdate
        use_optimal_control: Final = UseOptimalControlUpdate


class padding:
    class numpy:
        zero: Final = NumPyZeroPadding

    class jax:
        zero: Final = JaxZeroPadding


class filters:
    class numpy:
        none: Final = NoFilter
        savgol: Final = NumPySavGolFilter.create

    class jax:
        none: Final = NoFilter
        savgol: Final = JaxSavGolFilter.create
