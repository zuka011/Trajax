from typing import Protocol

from trajax.types.array import DataType

from numtypes import Array, Dims


class NumPyBicycleObstacleStatesHistory[T: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, int, K]]:
        """Returns the state history as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, K]]:
        """Returns the x positions of obstacles over time."""
        ...

    def y(self) -> Array[Dims[T, K]]:
        """Returns the y positions of obstacles over time."""
        ...

    def heading(self) -> Array[Dims[T, K]]:
        """Returns the headings of obstacles over time."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the state history."""
        ...

    @property
    def count(self) -> K:
        """Number of states."""
        ...
