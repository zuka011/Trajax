from typing import Protocol

from trajax.types.predictors import ObstacleStatesHistory
from trajax.types.models.integrator.common import (
    IntegratorState,
    IntegratorStateSequence,
    IntegratorStateBatch,
    IntegratorControlInputSequence,
    IntegratorControlInputBatch,
)

from jaxtyping import Array as JaxArray, Float


class JaxIntegratorState[D_x: int](IntegratorState[D_x], Protocol):
    @property
    def array(self) -> Float[JaxArray, "D_x"]:
        """Returns the underlying JAX array representing the integrator state."""
        ...


class JaxIntegratorStateSequence[T: int, D_x: int](
    IntegratorStateSequence[T, D_x], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_x"]:
        """Returns the underlying JAX array representing the integrator state sequence."""
        ...


class JaxIntegratorStateBatch[T: int, D_x: int, M: int](
    IntegratorStateBatch[T, D_x, M], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_x M"]:
        """Returns the underlying JAX array representing the integrator state batch."""
        ...


class JaxIntegratorControlInputSequence[T: int, D_u: int](
    IntegratorControlInputSequence[T, D_u], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        """Returns the underlying JAX array representing the integrator control input sequence."""
        ...


class JaxIntegratorControlInputBatch[T: int, D_u: int, M: int](
    IntegratorControlInputBatch[T, D_u, M], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_u M"]:
        """Returns the underlying JAX array representing the integrator control input batch."""
        ...


class JaxIntegratorObstacleStatesHistory[T: int, D_o: int, K: int](
    ObstacleStatesHistory[T, D_o, K], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the obstacle history as a JAX array."""
        ...
