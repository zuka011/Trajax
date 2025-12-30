from typing import Protocol

from numtypes import Array, Dims, D


type NumPyInitialPositionCovariance[K: int] = Array[Dims[D[2], D[2], K]]
type NumPyInitialVelocityCovariance[K: int] = Array[Dims[D[2], D[2], K]]
type NumPyPositionCovariance[T: int, K: int] = Array[Dims[T, D[2], D[2], K]]


class NumPyInitialCovarianceProvider[StateSequencesT](Protocol):
    def position(self, states: StateSequencesT) -> NumPyInitialPositionCovariance:
        """Provides the initial position covariance for the given obstacle states."""
        ...

    def velocity(self, states: StateSequencesT) -> NumPyInitialVelocityCovariance:
        """Provides the initial velocity covariance for the given obstacle states."""
        ...
