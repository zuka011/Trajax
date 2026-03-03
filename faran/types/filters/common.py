from typing import Any, Protocol

from faran.types.array import ArrayConvertible


class Noise(Protocol):
    @property
    def process_noise_covariance(self) -> ArrayConvertible:
        """Returns the process noise covariance matrix."""
        ...

    @property
    def observation_noise_covariance(self) -> ArrayConvertible:
        """Returns the observation noise covariance matrix."""
        ...


class NoiseModel[NoiseT, BeliefT, ObservationT, StateT = Any](Protocol):
    def __call__(
        self,
        *,
        noise: NoiseT,
        prediction: BeliefT,
        observation: ObservationT,
        state: StateT,
    ) -> tuple[NoiseT, StateT]:
        """Returns the updated noise covariances and updated state."""
        ...

    @property
    def state(self) -> StateT:
        """Returns the initial state for this noise model."""
        ...


class NoiseModelProvider[NoiseT, BeliefT, ObservationT, MatrixT, StateT = Any](
    Protocol
):
    def __call__(
        self, *, observation_matrix: MatrixT, noise: NoiseT
    ) -> NoiseModel[NoiseT, BeliefT, ObservationT, StateT]:
        """Creates a noise model using the specified observation matrix and
        initial noise covariances."""
        ...
