from typing import Protocol, TypedDict, Mapping

from trajax.types.array import DataType

from numtypes import Array, Dims, D


type BoundaryPoints[L: int = int] = Array[Dims[L, D[2]]]
type Breakpoint = float
type BoundaryWidthsDescription = Mapping[Breakpoint, "BoundaryWidths"]
"""A mapping describing the widths of the boundary at various segments
along a reference trajectory.

Example:
    If the corridor width changes at longitudinal positions 0.0, 10.0, and 20.0
    along the reference trajectory, the mapping could be defined as:
    {
        0.0: {"left": 2.0, "right": 2.0},
        10.0: {"left": 1.5, "right": 2.5},
        20.0: {"left": 2.0, "right": 1.0},
    }
"""


class BoundaryWidths(TypedDict):
    left: float
    """Distance to the left boundary at a breakpoint."""

    right: float
    """Distance to the right boundary at a breakpoint."""


class BoundaryDistance[T: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        """Returns the distances between the ego and the nearest boundary as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon over which the distances are defined."""
        ...

    @property
    def rollout_count(self) -> M:
        """The number of rollouts for which the distances are defined."""
        ...


class BoundaryDistanceExtractor[StateBatchT, DistanceT](Protocol):
    def __call__(self, *, states: StateBatchT) -> DistanceT:
        """Computes the distances between the ego and the nearest boundary."""
        ...


class ExplicitBoundary(Protocol):
    def left[L: int](self, *, sample_count: L = 100) -> BoundaryPoints[L]:
        """Returns the left boundary points as a NumPy array."""
        ...

    def right[L: int](self, *, sample_count: L = 100) -> BoundaryPoints[L]:
        """Returns the right boundary points as a NumPy array."""
        ...
