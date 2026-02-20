from typing import Protocol, Any

from faran.types.predictors import ObstacleStatesHistory

from numtypes import Array, Dims


class NumPyBicycleObstacleStatesHistory[T: int, K: int](
    ObstacleStatesHistory[T, Any, K], Protocol
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

    @property
    def array(self) -> Array[Dims[T, Any, K]]:
        """Returns the obstacle history as a NumPy array."""
        ...
