from typing import Self, Sequence, Protocol
from dataclasses import dataclass

from faran.types import (
    DataType,
    Array,
    jaxtyped,
    ControlInputSequence,
    NumPyState,
    NumPyStateSequence,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyCosts,
    NumPySampledObstacleStates,
    NumPyObstacleStates,
    NumPyObstacleStatesForTimeStep,
)

import numpy as np
from jaxtyping import Float


class NumPyControlInputSequenceLike(ControlInputSequence, Protocol):
    """Protocol for NumPy control input sequences exposing an array property."""

    @property
    def array(self) -> Float[Array, "T D_u"]:
        """Returns the underlying NumPy array representing the control input sequence."""
        ...


@jaxtyped
@dataclass(frozen=True)
class NumPySimpleState(NumPyState):
    """NumPy simple flat state vector."""

    _array: Float[Array, " D_x"]

    @staticmethod
    def create(*, array: Float[Array, " D_x"]) -> "NumPySimpleState":
        """Creates a NumPy simple state from the given array."""
        return NumPySimpleState(array)

    @staticmethod
    def zeroes(*, dimension: int) -> "NumPySimpleState":
        """Creates a zeroed simple state for the given dimension."""
        return NumPySimpleState(np.zeros((dimension,)))

    def __array__(self, dtype: DataType | None = None) -> Float[Array, " D_x"]:
        return self.array

    @property
    def dimension(self) -> int:
        return self.array.shape[0]

    @property
    def array(self) -> Float[Array, " D_x"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPySimpleStateSequence(NumPyStateSequence):
    """NumPy state sequence as a 2D array (time × dimension)."""

    _array: Float[Array, "T D_x"]

    @staticmethod
    def of_states(states: Sequence[NumPyState]) -> "NumPySimpleStateSequence":
        """Creates a simple state sequence from a sequence of simple states."""
        assert len(states) > 0, "States sequence must not be empty."

        array = np.stack([state.array for state in states], axis=0)

        return NumPySimpleStateSequence(array)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_x"]:
        return self.array

    def batched(self) -> "NumPySimpleStateBatch":
        return NumPySimpleStateBatch.wrap(self.array[..., np.newaxis])

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "T D_x"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPySimpleStateBatch(NumPyStateBatch):
    """NumPy state batch as a 3D array (time × dimension × rollouts)."""

    _array: Float[Array, "T D_x M"]

    @staticmethod
    def wrap(
        array: Float[Array, "T D_x M"],
    ) -> "NumPySimpleStateBatch":
        """Creates a NumPy simple state batch from the given array."""
        return NumPySimpleStateBatch(array)

    @staticmethod
    def of_states(states: Sequence[NumPySimpleState]) -> "NumPySimpleStateBatch":
        """Creates a simple state batch from a sequence of simple states."""
        assert len(states) > 0, "States sequence must not be empty."

        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        return NumPySimpleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_x M"]:
        return self.array

    def rollout(self, index: int) -> NumPySimpleStateSequence:
        return NumPySimpleStateSequence(self.array[..., index])

    def at(self, *, time_step: int, rollout: int) -> NumPySimpleState:
        return NumPySimpleState(self.array[time_step, :, rollout])

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> int:
        return self.array.shape[1]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[2]

    @property
    def array(self) -> Float[Array, "T D_x M"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPySimpleControlInputSequence(NumPyControlInputSequence):
    """NumPy control input sequence as a 2D array (time × dimension)."""

    _array: Float[Array, "T D_u"]

    @staticmethod
    def create(*, array: Float[Array, "T D_u"]) -> "NumPySimpleControlInputSequence":
        """Creates a NumPy simple control input sequence from the given array."""
        return NumPySimpleControlInputSequence(array)

    @staticmethod
    def constant(
        value: Float[Array, " D_u"], *, horizon: int
    ) -> "NumPySimpleControlInputSequence":
        """Creates a constant control input sequence for the given horizon."""
        array = np.tile(value.reshape(1, -1), (horizon, 1))

        return NumPySimpleControlInputSequence(array)

    @staticmethod
    def zeroes(horizon: int, dimension: int) -> "NumPySimpleControlInputSequence":
        """Creates a zeroed control input sequence for the given horizon."""
        return NumPySimpleControlInputSequence.constant(
            np.zeros((dimension,)), horizon=horizon
        )

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_u"]:
        return self.array

    def similar(self, *, array: Float[Array, "L D_u"]) -> Self:
        return self.__class__(array)

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "T D_u"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPySimpleControlInputBatch(NumPyControlInputBatch):
    """NumPy control input batch as a 3D array (time × dimension × rollouts)."""

    _array: Float[Array, "T D_u M"]

    @staticmethod
    def zero(
        *, horizon: int, dimension: int, rollout_count: int = 1
    ) -> "NumPySimpleControlInputBatch":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        array = np.zeros((horizon, dimension, rollout_count))

        return NumPySimpleControlInputBatch(array)

    @staticmethod
    def create(*, array: Float[Array, "T D_u M"]) -> "NumPySimpleControlInputBatch":
        """Creates a NumPy simple control input batch from the given array."""
        return NumPySimpleControlInputBatch(array)

    @staticmethod
    def of(sequence: NumPyControlInputSequenceLike) -> "NumPySimpleControlInputBatch":
        """Creates a simple control input batch from a single control input sequence."""
        return NumPySimpleControlInputBatch(sequence.array[..., np.newaxis])

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_u M"]:
        return self.array

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> int:
        return self.array.shape[1]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[2]

    @property
    def array(self) -> Float[Array, "T D_u M"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPySimpleCosts(NumPyCosts):
    """NumPy cost values as a 2D array (time × rollouts)."""

    _array: Float[Array, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T M"]:
        return self.array

    def similar(self, *, array: Float[Array, "T M"]) -> Self:
        return self.__class__(array)

    def zero(self) -> Self:
        return self.__class__(np.zeros_like(self.array))

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "T M"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPySimpleSampledObstacleStates(NumPySampledObstacleStates):
    """NumPy sampled obstacle states as a 4D array (time × dimension × obstacles × samples)."""

    _array: Float[Array, "T D_o K N"]

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_o K N"]:
        return self.array

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> int:
        return self.array.shape[1]

    @property
    def count(self) -> int:
        return self.array.shape[2]

    @property
    def sample_count(self) -> int:
        return self.array.shape[3]

    @property
    def array(self) -> Float[Array, "T D_o K N"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPySimpleObstacleStatesForTimeStep(
    NumPyObstacleStatesForTimeStep["NumPySimpleObstacleStates"]
):
    """NumPy obstacle states for a single time step as a 2D array (dimension × obstacles)."""

    _array: Float[Array, "D_o K"]

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "D_o K"]:
        return self.array

    def replicate(self, *, horizon: int) -> "NumPySimpleObstacleStates":
        raise NotImplementedError(
            "Currently cannot be triggered by any component or test. Please "
            "implement once this functionality is actually needed."
        )

    @property
    def dimension(self) -> int:
        return self.array.shape[0]

    @property
    def count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[Array, "D_o K"]:
        return self._array


@jaxtyped
@dataclass(frozen=True)
class NumPySimpleObstacleStates(NumPyObstacleStates[NumPySimpleSampledObstacleStates]):
    """NumPy obstacle states as a 3D array (time × dimension × obstacles)."""

    _array: Float[Array, "T D_o K"]
    _covariance: Float[Array, "T D_o D_o K"] | None = None

    @staticmethod
    def create(
        *,
        states: Float[Array, "T D_o K"],
        covariance: Float[Array, "T D_o D_o K"] | None = None,
    ) -> "NumPySimpleObstacleStates":
        return NumPySimpleObstacleStates(states, covariance)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_o K"]:
        return self.array

    def single(self) -> NumPySimpleSampledObstacleStates:
        return NumPySimpleSampledObstacleStates(self.array[..., np.newaxis])

    def last(self) -> NumPySimpleObstacleStatesForTimeStep:
        return NumPySimpleObstacleStatesForTimeStep(self.array[-1, :, :])

    def covariance(self) -> Float[Array, "T D_o D_o K"] | None:
        return self._covariance

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> int:
        return self.array.shape[1]

    @property
    def count(self) -> int:
        return self.array.shape[2]

    @property
    def array(self) -> Float[Array, "T D_o K"]:
        return self._array
