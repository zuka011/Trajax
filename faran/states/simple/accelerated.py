from typing import cast, Self, overload, Sequence, Protocol
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    DataType,
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
from numtypes import Array, Dims, D

import numpy as np
import jax.numpy as jnp


class JaxControlInputSequenceLike[T: int, D_u: int](
    ControlInputSequence[T, D_u], Protocol
):
    """Protocol for JAX control input sequences exposing an array property."""

    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        """Returns the underlying JAX array representing the control input sequence."""
        ...


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleState[D_x: int](JaxState[D_x]):
    """JAX simple flat state vector."""

    _array: Float[JaxArray, "D_x"]

    @staticmethod
    def zeroes[D_x_: int](*, dimension: D_x_) -> "JaxSimpleState[D_x_]":
        """Creates a zeroed simple state for the given dimension."""
        return JaxSimpleState(jnp.zeros((dimension,)))

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return self._numpy_array

    @property
    def dimension(self) -> D_x:
        return cast(D_x, self.array.shape[0])

    @property
    def array(self) -> Float[JaxArray, "D_x"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[D_x]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleStateSequence[T: int, D_x: int](JaxStateSequence[T, D_x]):
    """JAX state sequence as a 2D array (time × dimension)."""

    _array: Float[JaxArray, "T D_x"]

    @staticmethod
    def of_states[T_: int, D_x_: int](
        states: Sequence[JaxSimpleState[D_x_]], *, horizon: T_ | None = None
    ) -> "JaxSimpleStateSequence[T_, D_x_]":
        """Creates a simple state sequence from a sequence of simple states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        dimension = states[0].dimension
        array = jnp.stack([state.array for state in states], axis=0)

        assert array.shape == (horizon, dimension), (
            f"Expected array shape {(horizon, dimension)}, but got {array.shape}"
        )

        return JaxSimpleStateSequence(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x]]:
        return self._numpy_array

    def batched(self) -> "JaxSimpleStateBatch[T, D_x, D[1]]":
        return JaxSimpleStateBatch.wrap(self.array[..., jnp.newaxis])

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_x:
        return cast(D_x, self.array.shape[1])

    @property
    def array(self) -> Float[JaxArray, "T D_x"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_x]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleStateBatch[T: int, D_x: int, M: int](JaxStateBatch[T, D_x, M]):
    """JAX state batch as a 3D array (time × dimension × rollouts)."""

    _array: Float[JaxArray, "T D_x M"]

    @staticmethod
    def wrap(array: Float[JaxArray, "T D_x M"]) -> "JaxSimpleStateBatch[T, D_x, M]":
        """Creates a JAX simple state batch from the given array."""
        return JaxSimpleStateBatch(array)

    @staticmethod
    def of_states[D_x_: int, T_: int = int](
        states: Sequence[JaxState[D_x_]], *, horizon: T_ | None = None
    ) -> "JaxSimpleStateBatch[T_, D_x_, int]":
        """Creates a simple state batch from a sequence of simple states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        dimension = states[0].dimension
        array = jnp.stack([state.array for state in states], axis=0)[:, :, jnp.newaxis]

        assert array.shape == (horizon, dimension, 1), (
            f"Expected array shape {(horizon, dimension, 1)}, but got {array.shape}"
        )

        return JaxSimpleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        return self._numpy_array

    def rollout(self, index: int) -> JaxSimpleStateSequence[T, D_x]:
        return JaxSimpleStateSequence(self.array[..., index])

    def at(self, *, time_step: int, rollout: int) -> JaxSimpleState[D_x]:
        return JaxSimpleState(self.array[time_step, :, rollout])

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_x:
        return cast(D_x, self.array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])

    @property
    def array(self) -> Float[JaxArray, "T D_x M"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_x, M]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleControlInputSequence[T: int, D_u: int](JaxControlInputSequence[T, D_u]):
    """JAX control input sequence as a 2D array (time × dimension)."""

    _array: Float[JaxArray, "T D_u"]

    @staticmethod
    def constant[T_: int, D_u_: int](
        value: Float[JaxArray, "D_u"] | Array[Dims[D_u_]], *, horizon: T_
    ) -> "JaxSimpleControlInputSequence[T_, D_u_]":
        """Creates a constant control input sequence for the given horizon and dimension."""
        return JaxSimpleControlInputSequence(
            jnp.tile(jnp.asarray(value)[jnp.newaxis, :], (horizon, 1))
        )

    @staticmethod
    def zeroes[T_: int, D_u_: int](
        horizon: T_, dimension: D_u_
    ) -> "JaxSimpleControlInputSequence[T_, D_u_]":
        """Creates a zeroed control input sequence for the given horizon."""
        return JaxSimpleControlInputSequence.constant(
            jnp.zeros((dimension,)), horizon=horizon
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return self._numpy_array

    @overload
    def similar(self, *, array: Float[JaxArray, "T D_u"]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L
    ) -> "JaxSimpleControlInputSequence[L, D_u]": ...

    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L | None = None
    ) -> "Self | JaxSimpleControlInputSequence[L, D_u]":
        length = length if length is not None else cast(L, array.shape[0])

        assert array.shape[0] == length, (
            f"Expected array with time horizon {length}, but got {array.shape[0]}"
        )

        return self.__class__(array)

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_u:
        return cast(D_u, self.array.shape[1])

    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_u]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleControlInputBatch[T: int, D_u: int, M: int](
    JaxControlInputBatch[T, D_u, M]
):
    """JAX control input batch as a 3D array (time × dimension × rollouts)."""

    _array: Float[JaxArray, "T D_u M"]

    @staticmethod
    def create(*, array: Float[JaxArray, "T D_u M"]) -> "JaxSimpleControlInputBatch":
        """Factory method to create a JAX simple control input batch from an array."""
        return JaxSimpleControlInputBatch(array)

    @staticmethod
    def zero[T_: int, D_u_: int, M_: int](
        *, horizon: T_, dimension: D_u_, rollout_count: M_ = 1
    ) -> "JaxSimpleControlInputBatch[T_, D_u_, M_]":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        return JaxSimpleControlInputBatch(
            jnp.zeros((horizon, dimension, rollout_count))
        )

    @staticmethod
    def of[T_: int, D_u_: int](
        sequence: JaxControlInputSequenceLike[T_, D_u_],
    ) -> "JaxSimpleControlInputBatch[T_, D_u_, D[1]]":
        """Creates a JAX simple control input batch from a single control input sequence."""
        array = sequence.array[..., jnp.newaxis]

        assert array.shape == (sequence.horizon, sequence.dimension, 1), (
            f"Expected array shape {(sequence.horizon, sequence.dimension, 1)}, "
            f"but got {array.shape}"
        )

        return JaxSimpleControlInputBatch(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        return self._numpy_array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_u:
        return cast(D_u, self.array.shape[1])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[2])

    @property
    def array(self) -> Float[JaxArray, "T D_u M"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_u, M]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleCosts[T: int, M: int](JaxCosts[T, M]):
    """JAX cost values as a 2D array (time × rollouts)."""

    _array: Float[JaxArray, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return self._numpy_array

    def similar(self, *, array: Float[JaxArray, "T M"]) -> Self:
        return self.__class__(array)

    def zero(self) -> Self:
        return self.__class__(jnp.zeros_like(self.array))

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[1])

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleSampledObstacleStates[T: int, D_o: int, K: int, N: int](
    JaxSampledObstacleStates[T, D_o, K, N]
):
    """JAX sampled obstacle states as a 4D array (time × dimension × obstacles × samples)."""

    _array: Float[JaxArray, "T D_o K N"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        return self._numpy_array

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[1])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[2])

    @property
    def sample_count(self) -> N:
        return cast(N, self.array.shape[3])

    @property
    def array(self) -> Float[JaxArray, "T D_o K N"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_o, K, N]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleObstacleStatesForTimeStep[D_o: int, K: int](
    JaxObstacleStatesForTimeStep[
        D_o, K, "JaxSimpleObstacleStates", NumPySimpleObstacleStatesForTimeStep
    ]
):
    """JAX obstacle states for a single time step as a 2D array (dimension × obstacles)."""

    _array: Float[JaxArray, "D_o K"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_o, K]]:
        return self._numpy_array

    def replicate[T: int](self, *, horizon: T) -> "JaxSimpleObstacleStates[T, D_o, K]":
        # TODO: Implement replicate method
        raise NotImplementedError()

    def numpy(self) -> NumPySimpleObstacleStatesForTimeStep[D_o, K]:
        return NumPySimpleObstacleStatesForTimeStep(self._numpy_array)

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[0])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[1])

    @property
    def array(self) -> Float[JaxArray, "D_o K"]:
        return self._array

    @cached_property
    def _numpy_array(self) -> Array[Dims[D_o, K]]:
        return np.asarray(self.array)


@jaxtyped
@dataclass(frozen=True)
class JaxSimpleObstacleStates[T: int, D_o: int, K: int](
    JaxObstacleStates[T, D_o, K, JaxSimpleSampledObstacleStates]
):
    """JAX obstacle states as a 3D array (time × dimension × obstacles)."""

    _array: Float[JaxArray, "T D_o K"]
    _covariance: Float[JaxArray, "T D_o D_o K"] | None = None

    @staticmethod
    def create[T_: int, D_o_: int, K_: int](
        *,
        states: Float[JaxArray, "T D_o K"],
        covariance: Float[JaxArray, "T D_o D_o K"] | None = None,
        horizon: T_ | None = None,
        dimension: D_o_ | None = None,
        count: K_ | None = None,
    ) -> "JaxSimpleObstacleStates[T_, D_o_, K_]":
        horizon = horizon if horizon is not None else cast(T_, states.shape[0])
        dimension = dimension if dimension is not None else cast(D_o_, states.shape[1])
        count = count if count is not None else cast(K_, states.shape[2])

        assert states.shape == (horizon, dimension, count), (
            f"Expected states shape {(horizon, dimension, count)}, but got {states.shape}"
        )
        assert covariance is None or covariance.shape == (
            horizon,
            dimension,
            dimension,
            count,
        ), (
            f"Expected covariance shape {(horizon, dimension, dimension, count)}, "
            f"but got {covariance.shape}"
        )

        return JaxSimpleObstacleStates(states, covariance)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        return self._numpy_array

    def single(self) -> JaxSimpleSampledObstacleStates[T, D_o, K, D[1]]:
        return JaxSimpleSampledObstacleStates(self.array[..., jnp.newaxis])

    def last(self) -> JaxSimpleObstacleStatesForTimeStep[D_o, K]:
        return JaxSimpleObstacleStatesForTimeStep(self.array[-1])

    def covariance(self) -> Array[Dims[T, D_o, D_o, K]] | None:
        return self._numpy_covariance

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def dimension(self) -> D_o:
        return cast(D_o, self.array.shape[1])

    @property
    def count(self) -> K:
        return cast(K, self.array.shape[2])

    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        return self._array

    @property
    def covariance_array(self) -> Float[JaxArray, "T D_o D_o K"] | None:
        return self._covariance

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_o, K]]:
        return np.asarray(self.array)

    @cached_property
    def _numpy_covariance(self) -> Array[Dims[T, D_o, D_o, K]] | None:
        return np.asarray(self._covariance) if self._covariance is not None else None
