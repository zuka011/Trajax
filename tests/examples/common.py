from typing import Protocol

from trajax import ObstacleStateProvider


class SimulatingObstacleStateProvider[ObstacleStatesT](
    ObstacleStateProvider[ObstacleStatesT], Protocol
):
    def step(self) -> None:
        """Advances the internal state of the obstacle state provider."""
        ...
