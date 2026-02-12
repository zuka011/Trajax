from typing import Protocol

from trajax.types.costs import PoseD_o

from numtypes import Array, Dims, D

type NumPyCovariance[T: int, D_o: int, K: int] = Array[Dims[T, D_o, D_o, K]]
type NumPyInitialPositionCovariance[K: int] = Array[Dims[D[2], D[2], K]]
type NumPyInitialVelocityCovariance[K: int] = Array[Dims[D[2], D[2], K]]
type NumPyPoseCovariance[T: int, K: int] = Array[Dims[T, PoseD_o, PoseD_o, K]]


# TODO: Review!
class NumPyCovarianceProvider[
    StateSequencesT,
    StateCovarianceT,
    InputCovarianceT,
](Protocol):
    """Protocol for providing both state and input covariance matrices.

    The state covariance P_0 represents uncertainty about the current state.
    The input covariance Σ_u represents uncertainty about control inputs.
    """

    def state(self, states: StateSequencesT) -> StateCovarianceT:
        """Returns the initial state covariance P_0 for the given obstacle states."""
        ...

    def input(self, states: StateSequencesT) -> InputCovarianceT:
        """Returns the input covariance Σ_u representing uncertainty about control inputs."""
        ...
