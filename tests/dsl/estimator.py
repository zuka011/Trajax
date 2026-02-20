from typing import Any, Protocol

from faran import EstimatedObstacleStates

from numtypes import Array


class ComponentExtractor[StateT, InputT, CovarianceT = Any](Protocol):
    def __call__(
        self, result: EstimatedObstacleStates[StateT, InputT, CovarianceT]
    ) -> Array:
        """Extracts a specific component from the estimated states or inputs."""
        ...
