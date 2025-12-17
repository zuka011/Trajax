from dataclasses import dataclass

from trajax.trajectory.common import D_r, D_R

from numtypes import array, Array, Dims

import numpy as np


@dataclass(frozen=True)
class NumPyPathParameters[T: int, M: int]:
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
class NumPyPositions[T: int, M: int]:
    x: Array[Dims[T, M]]
    y: Array[Dims[T, M]]

    @staticmethod
    def create[T_: int, M_: int](
        *,
        x: Array[Dims[T_, M_]],
        y: Array[Dims[T_, M_]],
    ) -> "NumPyPositions[T_, M_]":
        """Creates a NumPy positions instance from x and y coordinate arrays."""
        return NumPyPositions(x=x, y=y)


@dataclass(frozen=True)
class NumPyReferencePoints[T: int, M: int]:
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
            array=array(
                np.stack([x, y, heading], axis=-1).transpose(0, 2, 1).tolist(),
                shape=(T, D_R, M),
            )
        )

    def __array__(self) -> Array[Dims[T, D_r, M]]:
        return self.array

    def x(self) -> Array[Dims[T, M]]:
        return self.array[:, 0]

    def y(self) -> Array[Dims[T, M]]:
        return self.array[:, 1]

    def heading(self) -> Array[Dims[T, M]]:
        return self.array[:, 2]
