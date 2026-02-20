from typing import Protocol

from faran.types.obstacles.common import ObstacleSimulator


class NumPyObstacleSimulator[ObstacleStatesForTimeStepT](
    ObstacleSimulator[ObstacleStatesForTimeStepT], Protocol
): ...
