from typing import Protocol

from trajax.types.array import DataType

from numtypes import Array, Dims


class IntegratorState[D_x: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        """Returns the state as a NumPy array."""
        ...

    @property
    def dimension(self) -> D_x:
        """Returns the dimension of the state."""
        ...


class IntegratorStateBatch[T: int, D_x: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        """Returns the states as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the state batch."""
        ...

    @property
    def dimension(self) -> D_x:
        """State dimension."""
        ...

    @property
    def rollout_count(self) -> M:
        """Number of rollouts in the batch."""
        ...


class IntegratorControlInputSequence[T: int, D_u: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        """Returns the control input sequence as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the control input sequence."""
        ...

    @property
    def dimension(self) -> D_u:
        """Control input dimension."""
        ...


class IntegratorControlInputBatch[T: int, D_u: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        """Returns the control inputs as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the control input batch."""
        ...

    @property
    def dimension(self) -> D_u:
        """Control input dimension."""
        ...

    @property
    def rollout_count(self) -> M:
        """Number of rollouts in the batch."""
        ...


class IntegratorObstacleStates[D_x: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x, K]]:
        """Returns the states as a NumPy array."""
        ...

    @property
    def dimension(self) -> D_x:
        """State dimension."""
        ...

    @property
    def count(self) -> K:
        """Number of states."""
        ...


class IntegratorObstacleStateSequences[StatesT, T: int, D_x: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, K]]:
        """Returns the sequence of states over time as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the state sequences."""
        ...

    @property
    def dimension(self) -> D_x:
        """State dimension."""
        ...

    @property
    def count(self) -> K:
        """Number of state sequences."""
        ...


class IntegratorObstacleVelocities[D_v: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_v, K]]:
        """Returns the velocities as a NumPy array."""
        ...

    @property
    def dimension(self) -> D_v:
        """Velocity dimension."""
        ...

    @property
    def count(self) -> K:
        """Number of velocities."""
        ...


class IntegratorObstacleControlInputSequences[T: int, D_u: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, K]]:
        """Returns the control input sequences as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the control input sequences."""
        ...

    @property
    def dimension(self) -> D_u:
        """Control input dimension."""
        ...

    @property
    def count(self) -> K:
        """Number of control input sequences."""
        ...
