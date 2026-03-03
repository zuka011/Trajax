from typing import Any, Protocol, NamedTuple, runtime_checkable

from faran.types.array import Array

from jaxtyping import Float, Array as JaxArray, Scalar

type JaxNoiseCovarianceArrayDescription = (
    Float[Array, "D_c D_c"]
    | Float[Array, " D_c"]
    | Float[JaxArray, "D_c D_c"]
    | Float[JaxArray, " D_c"]
)

type JaxNoiseCovarianceDescription = JaxNoiseCovarianceArrayDescription | Scalar | float


class JaxGaussianBelief(NamedTuple):
    mean: Float[JaxArray, "D_x K"]
    covariance: Float[JaxArray, "D_x D_x K"]


class JaxNoiseCovariances(NamedTuple):
    process_noise_covariance: Float[JaxArray, "D_x D_x"]
    observation_noise_covariance: Float[JaxArray, "D_z D_z"]


@runtime_checkable
class JaxNoiseModel[StateT = Any](Protocol):
    def __call__(
        self,
        *,
        noise: JaxNoiseCovariances,
        prediction: JaxGaussianBelief,
        observation: Float[JaxArray, "D_z K"],
        state: StateT,
    ) -> tuple[JaxNoiseCovariances, StateT]:
        """Returns the updated noise covariances and updated state."""
        ...

    @property
    def state(self) -> StateT:
        """Returns the initial state for this noise model."""
        ...


@runtime_checkable
class JaxNoiseModelProvider[StateT = Any](Protocol):
    def __call__(
        self,
        *,
        observation_matrix: Float[JaxArray, "D_z D_x"],
        noise: JaxNoiseCovariances,
    ) -> JaxNoiseModel[StateT]:
        """Creates a noise model using the specified observation matrix and
        initial noise covariances."""
        ...
