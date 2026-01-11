from dataclasses import dataclass
from functools import cached_property

from trajax.types.trajectories.common import (
    D_r,
    D_R,
    PathParameters,
    ReferencePoints,
    Positions,
    LateralPositions,
    LongitudinalPositions,
)

from numtypes import array, Array, Dims, D

import numpy as np


@dataclass(frozen=True)
class NumPyPathParameters[T: int, M: int](PathParameters[T, M]):
    array: Array[Dims[T, M]]

    def __array__(self) -> Array[Dims[T, M]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[1]


@dataclass(frozen=True)
class NumPyPositions[T: int, M: int](Positions[T, M]):
    _x: Array[Dims[T, M]]
    _y: Array[Dims[T, M]]

    @staticmethod
    def create[T_: int, M_: int](
        *,
        x: Array[Dims[T_, M_]],
        y: Array[Dims[T_, M_]],
    ) -> "NumPyPositions[T_, M_]":
        """Creates a NumPy positions instance from x and y coordinate arrays."""
        return NumPyPositions(_x=x, _y=y)

    def __array__(self) -> Array[Dims[T, D[2], M]]:
        return self.array

    def x(self) -> Array[Dims[T, M]]:
        return self._x

    def y(self) -> Array[Dims[T, M]]:
        return self._y

    @property
    def horizon(self) -> T:
        return self._x.shape[0]

    @property
    def rollout_count(self) -> M:
        return self._x.shape[1]

    @property
    def array(self) -> Array[Dims[T, D[2], M]]:
        return self._array

    @cached_property
    def _array(self) -> Array[Dims[T, D[2], M]]:
        return np.stack([self._x, self._y], axis=1)


@dataclass(frozen=True)
class NumPyHeadings[T: int, M: int]:
    heading: Array[Dims[T, M]]

    @staticmethod
    def create[T_: int, M_: int](
        *,
        heading: Array[Dims[T_, M_]],
    ) -> "NumPyHeadings[T_, M_]":
        """Creates a NumPy headings instance from an array of headings."""
        return NumPyHeadings(heading)


@dataclass(frozen=True)
class NumPyReferencePoints[T: int, M: int](ReferencePoints[T, M]):
    array: Array[Dims[T, D_r, M]]

    @staticmethod
    def create[T_: int, M_: int](
        *,
        x: Array[Dims[T_, M_]],
        y: Array[Dims[T_, M_]],
        heading: Array[Dims[T_, M_]],
    ) -> "NumPyReferencePoints[T_, M_]":
        """Creates a NumPy reference points instance from x, y, and heading arrays."""
        T, M = x.shape
        return NumPyReferencePoints(
            array=array(np.stack([x, y, heading], axis=1).tolist(), shape=(T, D_R, M))
        )

    def __array__(self) -> Array[Dims[T, D_r, M]]:
        return self.array

    def x(self) -> Array[Dims[T, M]]:
        return self.array[:, 0]

    def y(self) -> Array[Dims[T, M]]:
        return self.array[:, 1]

    def heading(self) -> Array[Dims[T, M]]:
        return self.array[:, 2]


@dataclass(frozen=True)
class NumPyLateralPositions[T: int, M: int](LateralPositions[T, M]):
    _array: Array[Dims[T, M]]

    @staticmethod
    def create[T_: int, M_: int](
        array: Array[Dims[T_, M_]],
    ) -> "NumPyLateralPositions[T_, M_]":
        """Creates a NumPy lateral positions instance from an array."""
        return NumPyLateralPositions(array)

    def __array__(self) -> Array[Dims[T, M]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[T, M]]:
        return self._array


@dataclass(frozen=True)
class NumPyLongitudinalPositions[T: int, M: int](LongitudinalPositions[T, M]):
    _array: Array[Dims[T, M]]

    @staticmethod
    def create[T_: int, M_: int](
        array: Array[Dims[T_, M_]],
    ) -> "NumPyLongitudinalPositions[T_, M_]":
        """Creates a NumPy longitudinal positions instance from an array."""
        return NumPyLongitudinalPositions(array)

    def __array__(self) -> Array[Dims[T, M]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[T, M]]:
        return self._array
