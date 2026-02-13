from typing import Protocol

from trajax import EstimatedObstacleStates

from numtypes import Array


class ComponentExtractor[StateT, InputT](Protocol):
    def __call__(self, result: EstimatedObstacleStates[StateT, InputT]) -> Array:
        """Extracts a specific component from the estimated states or inputs."""
        ...
