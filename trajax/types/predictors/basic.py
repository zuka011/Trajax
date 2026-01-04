from typing import Protocol

from trajax.types.predictors.common import ObstacleStatesHistory

from numtypes import Array, Dims


class NumPyObstacleStatesHistory[T: int, D_o: int, K: int](
    ObstacleStatesHistory[T, D_o, K], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
        """Returns the obstacle state history as a NumPy array."""
        ...
