from typing import Protocol

from faran.types.trajectories import JaxPathParameters, JaxPositions, JaxHeadings
from faran.types.costs.common import PositionExtractor

from jaxtyping import Array as JaxArray, Float


class JaxPathParameterExtractor[StateBatchT](Protocol):
    def __call__(self, states: StateBatchT, /) -> JaxPathParameters:
        """Extracts path parameters from a batch of states."""
        ...


class JaxPathVelocityExtractor[InputBatchT](Protocol):
    def __call__(self, inputs: InputBatchT, /) -> Float[JaxArray, "T M"]:
        """Extracts path velocities from a batch of control inputs."""
        ...


class JaxPositionExtractor[StateBatchT](
    PositionExtractor[StateBatchT, JaxPositions], Protocol
): ...


class JaxHeadingExtractor[StateBatchT](Protocol):
    def __call__(self, states: StateBatchT, /) -> JaxHeadings:
        """Extracts heading angles from a batch of states."""
        ...
