from typing import Protocol

from trajax.types.trajectories import NumPyPathParameters, NumPyPositions, NumPyHeadings

from numtypes import Array, Dim2


class NumPyPathParameterExtractor[StateBatchT](Protocol):
    def __call__(self, states: StateBatchT, /) -> NumPyPathParameters:
        """Extracts path parameters from a batch of states."""
        ...


class NumPyPathVelocityExtractor[InputBatchT](Protocol):
    def __call__(self, inputs: InputBatchT, /) -> Array[Dim2]:
        """Extracts path velocities from a batch of control inputs."""
        ...


class NumPyPositionExtractor[StateBatchT](Protocol):
    def __call__(self, states: StateBatchT, /) -> NumPyPositions:
        """Extracts (x, y) positions from a batch of states."""
        ...


class NumPyHeadingExtractor[StateBatchT](Protocol):
    def __call__(self, states: StateBatchT, /) -> NumPyHeadings:
        """Extracts heading angles from a batch of states."""
        ...
