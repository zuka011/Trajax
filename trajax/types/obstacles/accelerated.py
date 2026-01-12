from typing import Protocol

from trajax.types.obstacles.common import ObstacleSimulator


class JaxObstacleSimulator[ObstacleStatesForTimeStepT](
    ObstacleSimulator[ObstacleStatesForTimeStepT], Protocol
): ...
