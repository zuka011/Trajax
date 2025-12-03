from typing import Protocol

from trajax.type import DataType

from numtypes import Array, Dims


class State[D_x: int = int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        """Returns the state as a NumPy array."""
        ...


class StateBatch[T: int = int, D_x: int = int, M: int = int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
        """Returns the states as a NumPy array."""
        ...


class ControlInputBatch[T: int = int, D_u: int = int, M: int = int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        """Returns the control inputs as a NumPy array."""
        ...

    @property
    def time_horizon(self) -> T:
        """Time horizon of the control input batch."""
        ...

    @property
    def rollout_count(self) -> M:
        """Number of rollouts in the batch."""
        ...


class DynamicalModel[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](Protocol):
    async def simulate(
        self, inputs: ControlInputBatchT, initial_state: StateT
    ) -> StateBatchT:
        """Simulates the dynamical model over the given control inputs starting from the
        provided initial state."""
        ...
