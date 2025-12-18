from typing import Protocol, Self
from dataclasses import dataclass

from trajax.type import DataType

from numtypes import Array, Dims


class State[D_x: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        """Returns the state as a NumPy array."""
        ...

    @property
    def dimension(self) -> D_x:
        """Returns the dimension of the state."""
        ...


class StateBatch[T: int, D_x: int, M: int](Protocol):
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


class ControlInputSequence[T: int, D_u: int](Protocol):
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


class ControlInputBatch[T: int, D_u: int, M: int](Protocol):
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


class DynamicalModel[
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
](Protocol):
    async def simulate(
        self, inputs: ControlInputBatchT, initial_state: StateT
    ) -> StateBatchT:
        """Simulates the dynamical model over the given control inputs starting from the
        provided initial state."""
        ...

    async def step(self, input: ControlInputSequenceT, state: StateT) -> StateT:
        """Simulates a single time step of the dynamical model given the control input and current
        state."""
        ...


class Costs[T: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        """Returns the costs as a NumPy array."""
        ...

    def zero(self) -> Self:
        """Returns zero costs similar to this one."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the costs."""
        ...

    @property
    def rollout_count(self) -> M:
        """Number of rollouts the costs correspond to."""
        ...


class CostFunction[InputT: ControlInputBatch, StateT: StateBatch, CostsT: Costs](
    Protocol
):
    def __call__(self, *, inputs: InputT, states: StateT) -> CostsT:
        """Computes the cost for each time step and rollout."""
        ...


class Sampler[SequenceT: ControlInputSequence, BatchT: ControlInputBatch](Protocol):
    def sample(self, *, around: SequenceT) -> BatchT:
        """Samples a batch of control input sequences around the given nominal input."""
        ...

    @property
    def rollout_count(self) -> int:
        """Number of rollouts the sampler generates."""
        ...


@dataclass(frozen=True)
class Control[InputT: ControlInputSequence]:
    optimal: InputT
    nominal: InputT


class Mppi[StateT: State, ControlInputSequenceT: ControlInputSequence](Protocol):
    async def step(
        self,
        *,
        temperature: float,
        nominal_input: ControlInputSequenceT,
        initial_state: StateT,
    ) -> Control[ControlInputSequenceT]:
        """Runs one iteration of the MPPI algorithm to compute the next optimal and nominal
        control sequences.
        """
        ...


class UpdateFunction[ControlInputSequenceT: ControlInputSequence](Protocol):
    def __call__(
        self,
        *,
        nominal_input: ControlInputSequenceT,
        optimal_input: ControlInputSequenceT,
    ) -> ControlInputSequenceT:
        """Updates the nominal control input sequence based on the optimal control input
        sequence.
        """
        ...


class PaddingFunction[NominalT: ControlInputSequence, PaddingT: ControlInputSequence](
    Protocol
):
    def __call__(self, *, nominal_input: NominalT, padding_size: int) -> PaddingT:
        """Generates padding values for the shifted nominal control input sequence."""
        ...


class FilterFunction[ControlInputSequenceT: ControlInputSequence](Protocol):
    def __call__(
        self, *, optimal_input: ControlInputSequenceT
    ) -> ControlInputSequenceT:
        """Filters the optimal control input after it is computed."""
        ...


class NoUpdate:
    """Returns the nominal input unchanged."""

    def __call__[ControlInputSequenceT](
        self,
        *,
        nominal_input: ControlInputSequenceT,
        optimal_input: ControlInputSequenceT,
    ) -> ControlInputSequenceT:
        return nominal_input


class UseOptimalControlUpdate:
    """Sets the nominal input to the optimal input."""

    def __call__[ControlInputSequenceT](
        self,
        *,
        nominal_input: ControlInputSequenceT,
        optimal_input: ControlInputSequenceT,
    ) -> ControlInputSequenceT:
        return optimal_input


class NoFilter:
    """Returns the optimal input unchanged."""

    def __call__[ControlInputSequenceT](
        self, *, optimal_input: ControlInputSequenceT
    ) -> ControlInputSequenceT:
        return optimal_input
