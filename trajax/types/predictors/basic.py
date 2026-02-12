from typing import Protocol, Any

from trajax.types.predictors.common import (
    ObstacleStatesHistory,
    ObstacleStateSequences,
    ObstacleControlInputSequences,
)

from numtypes import Array, Dims


class NumPyObstacleStatesHistory[T: int, D_o: int, K: int, ObstacleStatesForTimeStepT](
    ObstacleStatesHistory[T, D_o, K, ObstacleStatesForTimeStepT], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
        """Returns the obstacle state history as a NumPy array."""
        ...


class NumPyObstacleStateSequences[T: int, D_o: int, K: int, SingleSampleT = Any](
    ObstacleStateSequences[T, D_o, K, SingleSampleT], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
        """Returns the obstacle state sequences as a NumPy array."""
        ...


class NumPyObstacleControlInputSequences[T: int, D_u: int, K: int](
    ObstacleControlInputSequences[T, D_u, K], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_u, K]]:
        """Returns the obstacle control input sequences as a NumPy array."""
        ...
