from typing import Protocol

from trajax.types.costs import POSE_D_O

from jaxtyping import Array as JaxArray, Float

type JaxCovariance[T: int, D_o: int, K: int] = Float[JaxArray, "T D_o D_o K"]
type JaxInitialPositionCovariance[K: int] = Float[JaxArray, "2 2 K"]
type JaxInitialVelocityCovariance[K: int] = Float[JaxArray, "2 2 K"]
type JaxPoseCovariance[T: int, K: int] = Float[JaxArray, f"T {POSE_D_O} {POSE_D_O} K"]


# TODO: Review!
class JaxCovarianceProvider[
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
