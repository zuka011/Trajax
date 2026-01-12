from typing import Protocol

from trajax.types.obstacles.common import ObstacleSimulator


class NumPyObstacleSimulator[ObstacleStatesForTimeStepT](
    ObstacleSimulator[ObstacleStatesForTimeStepT], Protocol
): ...
