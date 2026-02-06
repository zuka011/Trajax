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
    """Factory namespace for creating MPPI planners."""

    class numpy:
        base: Final = NumPyMppi.create
        augmented: Final = NumPyAugmentedMppi.create
        mpcc: Final = NumPyMpccMppi.create

    class jax:
        base: Final = JaxMppi.create
        augmented: Final = JaxAugmentedMppi.create
        mpcc: Final = JaxMpccMppi.create


class update:
    """Factory namespace for MPPI control sequence update strategies."""

    class numpy:
        none: Final = NoUpdate
        use_optimal_control: Final = UseOptimalControlUpdate

    class jax:
        none: Final = NoUpdate
        use_optimal_control: Final = UseOptimalControlUpdate


class padding:
    """Factory namespace for MPPI control sequence padding strategies."""

    class numpy:
        zero: Final = NumPyZeroPadding

    class jax:
        zero: Final = JaxZeroPadding


class filters:
    """Factory namespace for MPPI control sequence filtering strategies."""

    class numpy:
        none: Final = NoFilter
        savgol: Final = NumPySavGolFilter.create

    class jax:
        none: Final = NoFilter
        savgol: Final = JaxSavGolFilter.create
