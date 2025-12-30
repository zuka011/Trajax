from typing import Protocol

from jaxtyping import Array as JaxArray, Float


type JaxInitialPositionCovariance[K: int] = Float[JaxArray, "2 2 K"]
type JaxInitialVelocityCovariance[K: int] = Float[JaxArray, "2 2 K"]
type JaxPositionCovariance[T: int, K: int] = Float[JaxArray, "T 2 2 K"]


class JaxInitialCovarianceProvider[StateSequencesT](Protocol):
    def position(self, states: StateSequencesT) -> JaxInitialPositionCovariance:
        """Provides the initial position covariance for the given obstacle states."""
        ...

    def velocity(self, states: StateSequencesT) -> JaxInitialVelocityCovariance:
        """Provides the initial velocity covariance for the given obstacle states."""
        ...
