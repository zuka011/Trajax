from typing import Protocol

from trajax.type import DataType

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


class IntegratorModel[
    StateT: IntegratorState,
    StateBatchT: IntegratorStateBatch,
    ControlInputSequenceT: IntegratorControlInputSequence,
    ControlInputBatchT: IntegratorControlInputBatch,
](Protocol):
    def simulate(
        self, inputs: ControlInputBatchT, initial_state: StateT
    ) -> StateBatchT:
        """Simulates a single integrator model over the given control inputs starting from the
        initial state."""
        ...

    def step(self, input: ControlInputSequenceT, state: StateT) -> StateT:
        """Simulates a single time step of the integrator model given the control input and
        current state."""
        ...
