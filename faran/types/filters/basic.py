from typing import Any, Protocol, NamedTuple, runtime_checkable

from faran.types.array import Array

from jaxtyping import Float


type NumPyNoiseCovarianceArrayDescription = (
    Float[Array, "D_c D_c"] | Float[Array, " D_c"]
)

type NumPyNoiseCovarianceDescription = NumPyNoiseCovarianceArrayDescription | float


class NumPyGaussianBelief(NamedTuple):
    mean: Float[Array, "D_x K"]
    covariance: Float[Array, "D_x D_x K"]


class NumPyNoiseCovariances(NamedTuple):
    process_noise_covariance: Float[Array, "D_x D_x"]
    observation_noise_covariance: Float[Array, "D_z D_z"]


@runtime_checkable
class NumPyNoiseModel[StateT = Any](Protocol):
    def __call__(
        self,
        *,
        noise: NumPyNoiseCovariances,
        prediction: NumPyGaussianBelief,
        observation: Float[Array, "D_z K"],
        state: StateT,
    ) -> tuple[NumPyNoiseCovariances, StateT]:
        """Returns the updated noise covariances and updated state."""
        ...

    @property
    def state(self) -> StateT:
        """Returns the initial state for this noise model."""
        ...


@runtime_checkable
class NumPyNoiseModelProvider[StateT = Any](Protocol):
    def __call__(
        self,
        *,
        observation_matrix: Float[Array, "D_z D_x"],
        noise: NumPyNoiseCovariances,
    ) -> NumPyNoiseModel[StateT]:
        """Creates a noise model using the specified observation matrix and
        initial noise covariances."""
        ...
