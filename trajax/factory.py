from typing import Final

from trajax.mppi import (
    NumPyMppi,
    NumPyZeroPadding,
    JaxMppi,
    JaxZeroPadding,
    ControlCollector,
    NoUpdate,
    UseOptimalControlUpdate,
    NoFilter,
    NumPySavGolFilter,
    JaxSavGolFilter,
)
from trajax.states import AugmentedMppi, NumPyAugmentedMppi, JaxAugmentedMppi


class mppi:
    augmented: Final = AugmentedMppi.create

    class collector:
        controls: Final = ControlCollector

    class numpy:
        base: Final = NumPyMppi.create
        augmented: Final = NumPyAugmentedMppi.create

    class jax:
        base: Final = JaxMppi.create
        augmented: Final = JaxAugmentedMppi.create


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
