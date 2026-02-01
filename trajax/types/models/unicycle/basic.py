from typing import Protocol

from trajax.types.predictors import ObstacleStatesHistory
from trajax.types.models.unicycle.common import UnicycleD_o

from numtypes import Array, Dims


class NumPyUnicycleObstacleStatesHistory[T: int, K: int](
    ObstacleStatesHistory[T, UnicycleD_o, K], Protocol
):
    def x(self) -> Array[Dims[T, K]]:
        """Returns the x positions of the obstacles over time."""
        ...

    def y(self) -> Array[Dims[T, K]]:
        """Returns the y positions of the obstacles over time."""
        ...

    def heading(self) -> Array[Dims[T, K]]:
        """Returns the headings of the obstacles over time."""
        ...
