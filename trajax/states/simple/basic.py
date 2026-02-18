from typing import Self, overload, cast, Sequence, Protocol
from dataclasses import dataclass

from trajax.types import (
    DataType,
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
from numtypes import Array, Dims, D, shape_of


class NumPyControlInputSequenceLike[T: int, D_u: int](
    ControlInputSequence[T, D_u], Protocol
):
    """Protocol for NumPy control input sequences exposing an array property."""

    @property
    def array(self) -> Array[Dims[T, D_u]]:
        """Returns the underlying NumPy array representing the control input sequence."""
        ...


@dataclass(frozen=True)
class NumPySimpleState[D_x: int](NumPyState[D_x]):
    """NumPy simple flat state vector."""

    _array: Array[Dims[D_x]]

    @staticmethod
    def zeroes[D_x_: int](*, dimension: D_x_) -> "NumPySimpleState[D_x_]":
        """Creates a zeroed simple state for the given dimension."""
        array = np.zeros((dimension,))

        assert shape_of(array, matches=(dimension,))

        return NumPySimpleState(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        return self.array

    @property
    def dimension(self) -> D_x:
        return self.array.shape[0]

    @property
    def array(self) -> Array[Dims[D_x]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleStateSequence[T: int, D_x: int](NumPyStateSequence[T, D_x]):
    """NumPy state sequence as a 2D array (time × dimension)."""

    _array: Array[Dims[T, D_x]]

    @staticmethod
    def of_states[T_: int, D_x_: int](
        states: Sequence[NumPyState[D_x_]], *, horizon: T_ | None = None
    ) -> "NumPySimpleStateSequence[T_, D_x_]":
        """Creates a simple state sequence from a sequence of simple states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        dimension = states[0].dimension
        array = np.stack([state.array for state in states], axis=0)

        assert shape_of(array, matches=(horizon, dimension))

        return NumPySimpleStateSequence(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x]]:
        return self.array

    def batched(self) -> "NumPySimpleStateBatch[T, D_x, D[1]]":
        return NumPySimpleStateBatch.wrap(self.array[..., np.newaxis])

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_x:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[T, D_x]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleStateBatch[T: int, D_x: int, M: int](NumPyStateBatch[T, D_x, M]):
    """NumPy state batch as a 3D array (time × dimension × rollouts)."""

    _array: Array[Dims[T, D_x, M]]

    @staticmethod
    def wrap[T_: int, D_x_: int, M_: int](
        array: Array[Dims[T_, D_x_, M_]],
    ) -> "NumPySimpleStateBatch[T_, D_x_, M_]":
        """Creates a NumPy simple state batch from the given array."""
        return NumPySimpleStateBatch(array)

    @staticmethod
    def of_states[D_x_: int, T_: int = int](
        states: Sequence[NumPySimpleState[D_x_]], *, horizon: T_ | None = None
    ) -> "NumPySimpleStateBatch[T_, D_x_, D[1]]":
        """Creates a simple state batch from a sequence of simple states."""
        assert len(states) > 0, "States sequence must not be empty."

        horizon = horizon if horizon is not None else cast(T_, len(states))
        dimension = states[0].dimension
        array = np.stack([state.array for state in states], axis=0)[:, :, np.newaxis]

        assert shape_of(array, matches=(horizon, dimension, 1))

        return NumPySimpleStateBatch(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        return self.array

    def rollout(self, index: int) -> NumPySimpleStateSequence[T, D_x]:
        return NumPySimpleStateSequence(self.array[..., index])

    def at(self, *, time_step: int, rollout: int) -> NumPySimpleState[D_x]:
        return NumPySimpleState(self.array[time_step, :, rollout])

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_x:
        return self.array.shape[1]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, D_x, M]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleControlInputSequence[T: int, D_u: int](
    NumPyControlInputSequence[T, D_u]
):
    """NumPy control input sequence as a 2D array (time × dimension)."""

    _array: Array[Dims[T, D_u]]

    @staticmethod
    def constant[T_: int, D_u_: int](
        value: Array[Dims[D_u_]], *, horizon: T_
    ) -> "NumPySimpleControlInputSequence[T_, D_u_]":
        """Creates a constant control input sequence for the given horizon."""
        array = np.tile(value.reshape(1, -1), (horizon, 1))

        assert shape_of(array, matches=(horizon, value.shape[0]))

        return NumPySimpleControlInputSequence(array)

    @staticmethod
    def zeroes[T_: int, D_u_: int](
        horizon: T_, dimension: D_u_
    ) -> "NumPySimpleControlInputSequence[T_, D_u_]":
        """Creates a zeroed control input sequence for the given horizon."""
        zeros = np.zeros((dimension,))

        assert shape_of(zeros, matches=(dimension,))

        return NumPySimpleControlInputSequence.constant(zeros, horizon=horizon)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        return self.array

    @overload
    def similar(self, *, array: Array[Dims[T, D_u]]) -> Self: ...

    @overload
    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]], length: L
    ) -> "NumPySimpleControlInputSequence[L, D_u]": ...

    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]], length: L | None = None
    ) -> "Self | NumPySimpleControlInputSequence[L, D_u]":
        assert length is None or length == array.shape[0], (
            f"Length mismatch: expected {length}, got {array.shape[0]}"
        )

        # NOTE: "Wrong" cast to silence the type checker.
        return self.__class__(cast(Array[Dims[T, D_u]], array))

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_u:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[T, D_u]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleControlInputBatch[T: int, D_u: int, M: int](
    NumPyControlInputBatch[T, D_u, M]
):
    """NumPy control input batch as a 3D array (time × dimension × rollouts)."""

    _array: Array[Dims[T, D_u, M]]

    @staticmethod
    def zero[T_: int, D_u_: int, M_: int](
        *, horizon: T_, dimension: D_u_, rollout_count: M_ = 1
    ) -> "NumPySimpleControlInputBatch[T_, D_u_, M_]":
        """Creates a zeroed control input batch for the given horizon and rollout count."""
        array = np.zeros((horizon, dimension, rollout_count))

        assert shape_of(array, matches=(horizon, dimension, rollout_count))

        return NumPySimpleControlInputBatch(array)

    @staticmethod
    def create[T_: int, D_u_: int, M_: int](
        *, array: Array[Dims[T_, D_u_, M_]]
    ) -> "NumPySimpleControlInputBatch[T_, D_u_, M_]":
        """Creates a NumPy simple control input batch from the given array."""
        return NumPySimpleControlInputBatch(array)

    @staticmethod
    def of[T_: int, D_u_: int](
        sequence: NumPyControlInputSequenceLike[T_, D_u_],
    ) -> "NumPySimpleControlInputBatch[T_, D_u_, D[1]]":
        """Creates a simple control input batch from a single control input sequence."""
        array = sequence.array[..., np.newaxis]

        assert shape_of(array, matches=(sequence.horizon, sequence.dimension, 1))

        return NumPySimpleControlInputBatch(array)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_u:
        return self.array.shape[1]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, D_u, M]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleCosts[T: int, M: int](NumPyCosts[T, M]):
    """NumPy cost values as a 2D array (time × rollouts)."""

    _array: Array[Dims[T, M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return self.array

    def similar(self, *, array: Array[Dims[T, M]]) -> Self:
        return self.__class__(array)

    def zero(self) -> Self:
        return self.__class__(np.zeros_like(self.array))

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def rollout_count(self) -> M:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[T, M]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleSampledObstacleStates[T: int, D_o: int, K: int, N: int](
    NumPySampledObstacleStates[T, D_o, K, N]
):
    """NumPy sampled obstacle states as a 4D array (time × dimension × obstacles × samples)."""

    _array: Array[Dims[T, D_o, K, N]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        return self.array

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_o:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]

    @property
    def sample_count(self) -> N:
        return self.array.shape[3]

    @property
    def array(self) -> Array[Dims[T, D_o, K, N]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleObstacleStatesForTimeStep[D_o: int, K: int](
    NumPyObstacleStatesForTimeStep[D_o, K, "NumPySimpleObstacleStates"]
):
    """NumPy obstacle states for a single time step as a 2D array (dimension × obstacles)."""

    _array: Array[Dims[D_o, K]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_o, K]]:
        return self.array

    def replicate(self, *, horizon: int) -> "NumPySimpleObstacleStates":
        # TODO: Implement replicate method
        raise NotImplementedError()

    @property
    def dimension(self) -> D_o:
        return self.array.shape[0]

    @property
    def count(self) -> K:
        return self.array.shape[1]

    @property
    def array(self) -> Array[Dims[D_o, K]]:
        return self._array


@dataclass(frozen=True)
class NumPySimpleObstacleStates[T: int, D_o: int, K: int](
    NumPyObstacleStates[T, D_o, K, NumPySimpleSampledObstacleStates[T, D_o, K, D[1]]]
):
    """NumPy obstacle states as a 3D array (time × dimension × obstacles)."""

    _array: Array[Dims[T, D_o, K]]
    _covariance: Array[Dims[T, D_o, D_o, K]] | None = None

    @staticmethod
    def create[T_: int, D_o_: int, K_: int](
        *,
        states: Array[Dims[T_, D_o_, K_]],
        covariance: Array[Dims[T_, D_o_, D_o_, K_]] | None = None,
    ) -> "NumPySimpleObstacleStates[T_, D_o_, K_]":
        return NumPySimpleObstacleStates(states, covariance)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        return self.array

    def single(self) -> NumPySimpleSampledObstacleStates[T, D_o, K, D[1]]:
        return NumPySimpleSampledObstacleStates(self.array[..., np.newaxis])

    def last(self) -> NumPySimpleObstacleStatesForTimeStep[D_o, K]:
        return NumPySimpleObstacleStatesForTimeStep(self.array[-1, :, :])

    def covariance(self) -> Array[Dims[T, D_o, D_o, K]] | None:
        return self._covariance

    @property
    def horizon(self) -> T:
        return self.array.shape[0]

    @property
    def dimension(self) -> D_o:
        return self.array.shape[1]

    @property
    def count(self) -> K:
        return self.array.shape[2]

    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
        return self._array
