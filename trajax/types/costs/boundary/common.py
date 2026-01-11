from typing import Protocol

from trajax.types.array import DataType

from numtypes import Array, Dims


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
