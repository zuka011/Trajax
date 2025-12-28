from typing import Protocol

from trajax.types.predictors.common import ObstacleStatesHistory

from numtypes import Dims, Array


class NumPyObstacleStatesHistory[T: int, K: int](ObstacleStatesHistory[T, K], Protocol):
    def x(self) -> Array[Dims[T, K]]:
        """Returns the x-coordinates of the obstacle states over time."""
        ...

    def y(self) -> Array[Dims[T, K]]:
        """Returns the y-coordinates of the obstacle states over time."""
        ...

    def heading(self) -> Array[Dims[T, K]]:
        """Returns the headings of the obstacle states over time."""
        ...
