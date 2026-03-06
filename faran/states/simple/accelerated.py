from typing import Self, Sequence, Protocol
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    DataType,
    Array,
    jaxtyped,
    ControlInputSequence,
    JaxState,
    JaxStateSequence,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxCosts,
    JaxSampledObstacleStates,
    JaxObstacleStates,
    JaxObstacleStatesForTimeStep,
)
from faran.states.simple.basic import NumPySimpleObstacleStatesForTimeStep

from jaxtyping import Array as JaxArray, Float

import numpy as np
import jax.numpy as jnp


class JaxControlInputSequenceLike(ControlInputSequence, Protocol):
    """Protocol for JAX control input sequences exposing an array property."""

    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        """Returns the underlying JAX array representing the control input sequence."""
        ...


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleState(JaxState):
    """JAX simple flat state vector."""

    _array: Float[JaxArray, " D_x"]

    @staticmethod
    def create(
        *, array: Float[Array, " D_x"] | Float[JaxArray, " D_x"]
    ) -> "JaxSimpleState":
        """Creates a JAX simple state from the given array."""
        return JaxSimpleState(jnp.asarray(array))

    @staticmethod
    def zeroes(*, dimension: int) -> "JaxSimpleState":
        """Creates a zeroed simple state for the given dimension."""
        return JaxSimpleState(jnp.zeros((dimension,)))

    def __array__(self, dtype: DataType | None = None) -> Float[Array, " D_x"]:
        return self._numpy_array

    @property
    def dimension(self) -> int:
        return self.array.shape[0]

    @property
    def array(self) -> Float[JaxArray, " D_x"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, " D_x"]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleStateSequence(JaxStateSequence):
    """JAX state sequence as a 2D array (time × dimension)."""

    _array: Float[JaxArray, "T D_x"]

    @staticmethod
    def of_states(states: Sequence[JaxSimpleState]) -> "JaxSimpleStateSequence":
        """Creates a simple state sequence from a sequence of simple states."""
        assert len(states) > 0, "States sequence must not be empty."

        array = jnp.stack([state.array for state in states], axis=0)

        return JaxSimpleStateSequence(array)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_x"]:
        return self._numpy_array

    def batched(self) -> "JaxSimpleStateBatch":
        return JaxSimpleStateBatch.wrap(array=self.array[..., jnp.newaxis])

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[JaxArray, "T D_x"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "T D_x"]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleStateBatch(JaxStateBatch):
    """JAX state batch as a 3D array (time × dimension × rollouts)."""

    _array: Float[JaxArray, "T D_x M"]

    @staticmethod
    def wrap(
        *, array: Float[Array, "T D_x M"] | Float[JaxArray, "T D_x M"]
    ) -> "JaxSimpleStateBatch":
        """Creates a JAX simple state batch from the given array."""
        return JaxSimpleStateBatch(jnp.asarray(array))

    @staticmethod
    def of_states(states: Sequence[JaxState]) -> "JaxSimpleStateBatch":
        """Creates a simple state batch from a sequence of simple states."""
        assert len(states) > 0, "States sequence must not be empty."

        array = jnp.stack([state.array for state in states], axis=0)[:, :, jnp.newaxis]

        return JaxSimpleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_x M"]:
        return self._numpy_array

    def rollout(self, index: int) -> JaxSimpleStateSequence:
        return JaxSimpleStateSequence(self.array[..., index])

    def at(self, *, time_step: int, rollout: int) -> JaxSimpleState:
        return JaxSimpleState(self.array[time_step, :, rollout])

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
    def array(self) -> Float[JaxArray, "T D_x M"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "T D_x M"]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleControlInputSequence(JaxControlInputSequence):
    """JAX control input sequence as a 2D array (time × dimension)."""

    _array: Float[JaxArray, "T D_u"]

    @staticmethod
    def create(
        *, array: Float[Array, "T D_u"] | Float[JaxArray, "T D_u"]
    ) -> "JaxSimpleControlInputSequence":
        """Creates a JAX simple control input sequence from the given array."""
        return JaxSimpleControlInputSequence(jnp.asarray(array))

    @staticmethod
    def constant(
        value: Float[JaxArray, " D_u"] | Float[Array, " D_u"], *, horizon: int
    ) -> "JaxSimpleControlInputSequence":
        """Creates a constant control input sequence for the given horizon and dimension."""
        return JaxSimpleControlInputSequence(
            jnp.tile(jnp.asarray(value)[jnp.newaxis, :], (horizon, 1))
        )

    @staticmethod
    def zeroes(horizon: int, dimension: int) -> "JaxSimpleControlInputSequence":
        """Creates a zeroed control input sequence for the given horizon."""
        return JaxSimpleControlInputSequence.constant(
            jnp.zeros((dimension,)), horizon=horizon
        )

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_u"]:
        return self._numpy_array

    def similar(self, *, array: Float[JaxArray, "L D_u"]) -> Self:
        return self.__class__(array)

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def dimension(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "T D_u"]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleControlInputBatch(JaxControlInputBatch):
    """JAX control input batch as a 3D array (time × dimension × rollouts)."""

    _array: Float[JaxArray, "T D_u M"]

    @staticmethod
    def create(
        *, array: Float[Array, "T D_u M"] | Float[JaxArray, "T D_u M"]
    ) -> "JaxSimpleControlInputBatch":
        """Factory method to create a JAX simple control input batch from an array."""
        return JaxSimpleControlInputBatch(jnp.asarray(array))

    @staticmethod
    def zero(
        *, horizon: int, dimension: int, rollout_count: int = 1
    ) -> "JaxSimpleControlInputBatch":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        return JaxSimpleControlInputBatch(
            jnp.zeros((horizon, dimension, rollout_count))
        )

    @staticmethod
    def of(
        sequence: JaxControlInputSequenceLike,
    ) -> "JaxSimpleControlInputBatch":
        """Creates a JAX simple control input batch from a single control input sequence."""
        return JaxSimpleControlInputBatch(sequence.array[..., jnp.newaxis])

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_u M"]:
        return self._numpy_array

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
    def array(self) -> Float[JaxArray, "T D_u M"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "T D_u M"]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleCosts(JaxCosts):
    """JAX cost values as a 2D array (time × rollouts)."""

    _array: Float[JaxArray, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T M"]:
        return self._numpy_array

    def similar(self, *, array: Float[JaxArray, "T M"]) -> Self:
        return self.__class__(array)

    def zero(self) -> Self:
        return self.__class__(jnp.zeros_like(self.array))

    @property
    def horizon(self) -> int:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "T M"]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleSampledObstacleStates(JaxSampledObstacleStates):
    """JAX sampled obstacle states as a 4D array (time × dimension × obstacles × samples)."""

    _array: Float[JaxArray, "T D_o K N"]

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_o K N"]:
        return self._numpy_array

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
    def array(self) -> Float[JaxArray, "T D_o K N"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "T D_o K N"]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleObstacleStatesForTimeStep(
    JaxObstacleStatesForTimeStep[
        "JaxSimpleObstacleStates", NumPySimpleObstacleStatesForTimeStep
    ]
):
    """JAX obstacle states for a single time step as a 2D array (dimension × obstacles)."""

    _array: Float[JaxArray, "D_o K"]

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "D_o K"]:
        return self._numpy_array

    def replicate(self, *, horizon: int) -> "JaxSimpleObstacleStates":
        raise NotImplementedError(
            "Currently cannot be triggered by any component or test. Please "
            "implement once this functionality is actually needed."
        )

    def numpy(self) -> NumPySimpleObstacleStatesForTimeStep:
        return NumPySimpleObstacleStatesForTimeStep(self._numpy_array)

    @property
    def dimension(self) -> int:
        return self.array.shape[0]

    @property
    def count(self) -> int:
        return self.array.shape[1]

    @property
    def array(self) -> Float[JaxArray, "D_o K"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Float[Array, "D_o K"]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleObstacleStates(JaxObstacleStates[JaxSimpleSampledObstacleStates]):
    """JAX obstacle states as a 3D array (time × dimension × obstacles)."""

    _array: Float[JaxArray, "T D_o K"]
    _covariance: Float[JaxArray, "T D_o D_o K"] | None = None

    @staticmethod
    def create(
        *,
        states: Float[Array, "T D_o K"] | Float[JaxArray, "T D_o K"],
        covariance: Float[Array, "T D_o D_o K"]
        | Float[JaxArray, "T D_o D_o K"]
        | None = None,
    ) -> "JaxSimpleObstacleStates":

        return JaxSimpleObstacleStates(
            jnp.asarray(states),
            jnp.asarray(covariance) if covariance is not None else None,
        )

    def __array__(self, dtype: DataType | None = None) -> Float[Array, "T D_o K"]:
        return self._numpy_array

    def single(self) -> JaxSimpleSampledObstacleStates:
        return JaxSimpleSampledObstacleStates(self.array[..., jnp.newaxis])

    def last(self) -> JaxSimpleObstacleStatesForTimeStep:
        return JaxSimpleObstacleStatesForTimeStep(self.array[-1])

    def covariance(self) -> Float[Array, "T D_o D_o K"] | None:
        return self._numpy_covariance

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
    def array(self) -> Float[JaxArray, "T D_o K"]:
        return self._array

    @property
    def covariance_array(self) -> Float[JaxArray, "T D_o D_o K"] | None:
        return self._covariance

    @cached_property
    def _numpy_array(self) -> Float[Array, "T D_o K"]:
        return np.asarray(self.array)

    @cached_property
    def _numpy_covariance(self) -> Float[Array, "T D_o D_o K"] | None:
        return np.asarray(self._covariance) if self._covariance is not None else None
