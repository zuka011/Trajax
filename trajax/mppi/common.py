from typing import Protocol, Self
from dataclasses import dataclass

from trajax.type import DataType
from trajax.model import (
    DynamicalModel,
    State,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
)

from numtypes import Array, Dims


class Costs[T: int = int, M: int = int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        """Returns the control inputs as a NumPy array."""
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


class Sampler[SequenceT: ControlInputSequence, BatchT: ControlInputBatch, M: int = int](
    Protocol
):
    def sample(self, *, around: SequenceT) -> BatchT:
        """Samples a batch of control input sequences around the given nominal input."""
        ...

    @property
    def rollout_count(self) -> M:
        """Number of rollouts the sampler generates."""
        ...


@dataclass(frozen=True)
class Control[InputT: ControlInputSequence]:
    optimal: InputT
    nominal: InputT


class Mppi[
    InStateT: State,
    OutStateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,  # Invariant
    ControlInputBatchT: ControlInputBatch,
    CostsT: Costs,
](Protocol):
    async def step(
        self,
        *,
        model: DynamicalModel[
            InStateT, OutStateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
        ],
        cost_function: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
        sampler: Sampler[ControlInputSequenceT, ControlInputBatchT],
        temperature: float,
        nominal_input: ControlInputSequenceT,
        initial_state: InStateT,
    ) -> Control[ControlInputSequenceT]:
        """Runs one iteration of the MPPI algorithm to compute the next optimal and nominal
        control sequences.
        """
        ...


class UpdateFunction[
    ControlInputSequenceT: ControlInputSequence  # Invariant
](Protocol):
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


class FilterFunction[
    ControlInputSequenceT: ControlInputSequence  # Invariant
](Protocol):
    def __call__(
        self, *, optimal_input: ControlInputSequenceT
    ) -> ControlInputSequenceT:
        """Filters the optimal control input after it is computed."""
        ...


class NoUpdate[ControlInputSequenceT: ControlInputSequence]:
    """Returns the nominal input unchanged."""

    def __call__(
        self,
        *,
        nominal_input: ControlInputSequenceT,
        optimal_input: ControlInputSequenceT,
    ) -> ControlInputSequenceT:
        return nominal_input


class UseOptimalControlUpdate[ControlInputSequenceT: ControlInputSequence]:
    """Sets the nominal input to the optimal input."""

    def __call__(
        self,
        *,
        nominal_input: ControlInputSequenceT,
        optimal_input: ControlInputSequenceT,
    ) -> ControlInputSequenceT:
        return optimal_input


class NoFilter[ControlInputSequenceT: ControlInputSequence]:
    """Returns the optimal input unchanged."""

    def __call__(
        self, *, optimal_input: ControlInputSequenceT
    ) -> ControlInputSequenceT:
        return optimal_input
