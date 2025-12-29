from typing import Protocol

from trajax.types.predictors import ObstacleStatesHistory

from numtypes import Array, Dims, D


type D_o = D[3]


class NumPyBicycleObstacleStatesHistory[T: int, K: int](
    ObstacleStatesHistory[T, D_o, K], Protocol
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
