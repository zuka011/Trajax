from typing import Protocol

from faran.types.obstacles.common import ObstacleSimulator


class JaxObstacleSimulator[ObstacleStatesForTimeStepT](
    ObstacleSimulator[ObstacleStatesForTimeStepT], Protocol
): ...
