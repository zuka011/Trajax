from typing import Protocol, Final

from numtypes import Array, Dims, D

D_R: Final = 3

type D_r = D[3]


class PathParameters[T: int = int, M: int = int](Protocol):
    def __array__(self) -> Array[Dims[T, M]]:
        """Returns the path parameters as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """Returns the time horizon of the path parameters."""
        ...

    @property
    def rollout_count(self) -> M:
        """Returns the rollout count of the path parameters."""
        ...


class ReferencePoints[T: int = int, M: int = int](Protocol):
    def __array__(self) -> Array[Dims[T, D_r, M]]:
        """Returns the reference points as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, M]]:
        """Returns the x coordinates of the reference points."""
        ...

    def y(self) -> Array[Dims[T, M]]:
        """Returns the y coordinates of the reference points."""
        ...

    def heading(self) -> Array[Dims[T, M]]:
        """Returns the heading angles of the reference points."""
        ...


class Trajectory[PathParametersT: PathParameters, ReferencePointsT: ReferencePoints](
    Protocol
):
    def query(self, parameters: PathParametersT) -> ReferencePointsT:
        """Queries the trajectory at the given path parameters, returning the reference points."""
        ...
