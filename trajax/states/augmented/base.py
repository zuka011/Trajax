from typing import cast
from dataclasses import dataclass

from trajax.type import DataType
from trajax.model import ControlInputSequence
from trajax.states.augmented.common import AugmentedControlInputSequence
from numtypes import Array, Dims

import numpy as np


@dataclass(kw_only=True, frozen=True)
class BaseAugmentedControlInputSequence[
    PhysicalT: ControlInputSequence,
    VirtualT: ControlInputSequence,
    T: int = int,
    D_u: int = int,
](AugmentedControlInputSequence[PhysicalT, VirtualT, T, D_u]):
    _physical: PhysicalT
    _virtual: VirtualT
    _horizon: T
    _dimension: D_u

    @staticmethod
    def of[
        P: ControlInputSequence,
        V: ControlInputSequence,
        T_: int = int,
        D_u_: int = int,
    ](
        *,
        physical: P,
        virtual: V,
        horizon: T_ | None = None,
        dimension: D_u_ | None = None,
    ) -> "AugmentedControlInputSequence[P, V, T_, D_u_]":
        return BaseAugmentedControlInputSequence(
            _physical=physical,
            _virtual=virtual,
            _horizon=cast(
                T_,
                horizon if horizon is not None else physical.horizon,
            ),
            _dimension=cast(
                D_u_,
                dimension
                if dimension is not None
                else physical.dimension + virtual.dimension,
            ),
        )

    def __post_init__(self) -> None:
        assert self.horizon == self.physical.horizon == self.virtual.horizon, (
            f"Horizon mismatch in {self.__class__.__name__}: "
            f"expected {self.horizon}, got "
            f"{self.physical.horizon} (physical) and {self.virtual.horizon} (virtual)"
        )
        assert self.dimension == (self.physical.dimension + self.virtual.dimension), (
            f"Control dimension mismatch in {self.__class__.__name__}: "
            f"expected {self.dimension}, got "
            f"{self.physical.dimension} (physical) + {self.virtual.dimension} (virtual) "
            f"= {self.physical.dimension + self.virtual.dimension}"
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return np.concatenate(
            [
                np.asarray(self.physical, dtype=dtype),
                np.asarray(self.virtual, dtype=dtype),
            ],
            axis=1,
        )

    @property
    def physical(self) -> PhysicalT:
        return self._physical

    @property
    def virtual(self) -> VirtualT:
        return self._virtual

    @property
    def horizon(self) -> T:
        return self._horizon

    @property
    def dimension(self) -> D_u:
        return self._dimension
