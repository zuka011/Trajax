from typing import Protocol

from trajax.types.obstacles.history.common import (
    ObstaclePositionsForTimeStep,
    ObstaclePositions,
    ObstaclePositionExtractor,
)

from numtypes import Array, Dims


class NumPyObstaclePositionsForTimeStep[D_p: int, K: int](
    ObstaclePositionsForTimeStep[D_p, K], Protocol
):
    @property
    def array(self) -> Array[Dims[D_p, K]]:
        """Returns the obstacle positions for a single time step as a NumPy array."""
        ...


class NumPyObstaclePositions[T: int, D_p: int, K: int](
    ObstaclePositions[T, D_p, K], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_p, K]]:
        """Returns the obstacle positions as a NumPy array."""
        ...


class NumPyObstaclePositionExtractor[
    ObstacleStatesForTimeStepT,
    ObstacleStatesT,
    PositionsForTimeStepT,
    PositionsT,
](
    ObstaclePositionExtractor[
        ObstacleStatesForTimeStepT, ObstacleStatesT, PositionsForTimeStepT, PositionsT
    ],
    Protocol,
): ...
