from typing import Protocol

from trajax.types.trajectories import JaxPathParameters, JaxPositions, JaxHeadings

from jaxtyping import Array as JaxArray, Float


class JaxPathParameterExtractor[StateBatchT](Protocol):
    def __call__(self, states: StateBatchT, /) -> JaxPathParameters:
        """Extracts path parameters from a batch of states."""
        ...


class JaxPathVelocityExtractor[InputBatchT](Protocol):
    def __call__(self, inputs: InputBatchT, /) -> Float[JaxArray, "T M"]:
        """Extracts path velocities from a batch of control inputs."""
        ...


class JaxPositionExtractor[StateBatchT](Protocol):
    def __call__(self, states: StateBatchT, /) -> JaxPositions:
        """Extracts (x, y) positions from a batch of states."""
        ...


class JaxHeadingExtractor[StateBatchT](Protocol):
    def __call__(self, states: StateBatchT, /) -> JaxHeadings:
        """Extracts heading angles from a batch of states."""
        ...
