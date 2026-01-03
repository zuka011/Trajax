from typing import Protocol

from trajax.types.predictors import ObstacleStatesHistory
from trajax.types.models.integrator.common import (
    IntegratorState,
    IntegratorStateSequence,
    IntegratorStateBatch,
    IntegratorControlInputSequence,
    IntegratorControlInputBatch,
)

from numtypes import Array, Dims


class NumPyIntegratorState[D_x: int](IntegratorState[D_x], Protocol):
    @property
    def array(self) -> Array[Dims[D_x]]:
        """Returns the underlying NumPy array representing the integrator state."""
        ...


class NumPyIntegratorStateSequence[T: int, D_x: int](
    IntegratorStateSequence[T, D_x], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_x]]:
        """Returns the underlying NumPy array representing the integrator state sequence."""
        ...


class NumPyIntegratorStateBatch[T: int, D_x: int, M: int](
    IntegratorStateBatch[T, D_x, M], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_x, M]]:
        """Returns the underlying NumPy array representing the integrator state batch."""
        ...


class NumPyIntegratorControlInputSequence[T: int, D_u: int](
    IntegratorControlInputSequence[T, D_u], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_u]]:
        """Returns the underlying NumPy array representing the integrator control input
        sequence.
        """
        ...


class NumPyIntegratorControlInputBatch[T: int, D_u: int, M: int](
    IntegratorControlInputBatch[T, D_u, M], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_u, M]]:
        """Returns the underlying NumPy array representing the integrator control input
        batch.
        """
        ...


class NumPyIntegratorObstacleStatesHistory[T: int, D_o: int, K: int](
    ObstacleStatesHistory[T, D_o, K], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
        """Returns the obstacle history as a NumPy array."""
        ...
