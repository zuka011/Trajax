from typing import Protocol, Final

from numtypes import Array, Dims, D

D_R: Final = 3

type D_r = D[3]


class PathParameters[L: int, M: int](Protocol):
    def __array__(self) -> Array[Dims[L, M]]:
        """Returns the path parameters as a NumPy array."""
        ...

    @property
    def horizon(self) -> L:
        """Returns the time horizon of the path parameters."""
        ...

    @property
    def rollout_count(self) -> M:
        """Returns the rollout count of the path parameters."""
        ...


class ReferencePoints[L: int, M: int](Protocol):
    def __array__(self) -> Array[Dims[L, D_r, M]]:
        """Returns the reference points as a NumPy array."""
        ...


class Trajectory[PathParametersT: PathParameters, ReferencePointsT: ReferencePoints](
    Protocol
):
    def query(self, parameters: PathParametersT) -> ReferencePointsT:
        """Queries the trajectory at the given path parameters, returning the reference points."""
        ...
