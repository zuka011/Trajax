from typing import Protocol, Final, Any

from numtypes import Array, Dims, D

D_R: Final = 3

type D_r = D[3]
"""Dimensionality of reference points (x, y, heading)."""


class Positions[T: int, M: int](Protocol):
    def __array__(self) -> Array[Dims[T, D[2], M]]:
        """Returns the positions as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, M]]:
        """Returns the x coordinates of the positions."""
        ...

    def y(self) -> Array[Dims[T, M]]:
        """Returns the y coordinates of the positions."""
        ...

    @property
    def horizon(self) -> T:
        """Returns the time horizon of the positions."""
        ...

    @property
    def rollout_count(self) -> M:
        """Returns the rollout count of the positions."""
        ...


class LateralPositions[T: int, M: int](Protocol):
    def __array__(self) -> Array[Dims[T, M]]:
        """Returns the lateral positions as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """Returns the time horizon of the lateral positions."""
        ...

    @property
    def rollout_count(self) -> M:
        """Returns the rollout count of the lateral positions."""
        ...


class LongitudinalPositions[T: int, M: int](Protocol):
    def __array__(self) -> Array[Dims[T, M]]:
        """Returns the longitudinal positions as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """Returns the time horizon of the longitudinal positions."""
        ...

    @property
    def rollout_count(self) -> M:
        """Returns the rollout count of the longitudinal positions."""
        ...


class PathParameters[T: int, M: int](Protocol):
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


class ReferencePoints[T: int, M: int](Protocol):
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


class Trajectory[
    PathParametersT,
    ReferencePointsT,
    PositionsT = Any,
    LateralT = Any,
    LongitudinalT = Any,
](Protocol):
    def query(self, parameters: PathParametersT) -> ReferencePointsT:
        """Queries the trajectory at the given path parameters, returning the reference points."""
        ...

    def lateral(self, positions: PositionsT) -> LateralT:
        """Computes the lateral deviation from the trajectory for each position."""
        ...

    def longitudinal(self, positions: PositionsT) -> LongitudinalT:
        """Computes the longitudinal coordinate along the trajectory for each position."""
        ...

    @property
    def path_length(self) -> float:
        """Returns the total length of the reference trajectory."""
        ...
