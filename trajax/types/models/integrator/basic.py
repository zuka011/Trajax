from typing import Protocol

from trajax.types.models.integrator.common import (
    IntegratorState,
    IntegratorStateBatch,
    IntegratorControlInputSequence,
    IntegratorControlInputBatch,
    IntegratorObstacleStates,
    IntegratorObstacleStateSequences,
    IntegratorObstacleVelocities,
    IntegratorObstacleControlInputSequences,
)

from numtypes import Array, Dims


class NumPyIntegratorState[D_x: int](IntegratorState[D_x], Protocol):
    @property
    def array(self) -> Array[Dims[D_x]]:
        """Returns the underlying NumPy array representing the integrator state."""
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


class NumPyIntegratorObstacleStates[D_x: int, K: int](
    IntegratorObstacleStates[D_x, K], Protocol
):
    @property
    def array(self) -> Array[Dims[D_x, K]]:
        """Returns the underlying NumPy array representing the object states."""
        ...


class NumPyIntegratorObstacleStateSequences[T: int, D_x: int, K: int](
    IntegratorObstacleStateSequences[NumPyIntegratorObstacleStates[D_x, K], T, D_x, K],
    Protocol,
):
    @property
    def array(self) -> Array[Dims[T, D_x, K]]:
        """Returns the underlying NumPy array representing the object state sequences."""
        ...


class NumPyIntegratorObstacleVelocities[D_v: int, K: int](
    IntegratorObstacleVelocities[D_v, K], Protocol
):
    @property
    def array(self) -> Array[Dims[D_v, K]]:
        """Returns the underlying NumPy array representing the object velocities."""
        ...


class NumPyIntegratorObstacleControlInputSequences[T: int, D_u: int, K: int](
    IntegratorObstacleControlInputSequences[T, D_u, K], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_u, K]]:
        """Returns the underlying NumPy array representing the object control input
        sequences.
        """
        ...
