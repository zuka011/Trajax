from typing import Protocol
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


@dataclass(frozen=True)
class Control[InputT: ControlInputSequence]:
    optimal: InputT
    nominal: InputT


class Mppi[
    StateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
    CostsT: Costs,
](Protocol):
    async def step(
        self,
        *,
        model: DynamicalModel[
            ControlInputSequenceT, ControlInputBatchT, StateT, StateBatchT
        ],
        cost_function: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
        sampler: Sampler[ControlInputSequenceT, ControlInputBatchT],
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

    def __call__[ControlInputSequenceT: ControlInputSequence](
        self,
        *,
        nominal_input: ControlInputSequenceT,
        optimal_input: ControlInputSequenceT,
    ) -> ControlInputSequenceT:
        return nominal_input


class UseOptimalControlUpdate:
    """Sets the nominal input to the optimal input."""

    def __call__[ControlInputSequenceT: ControlInputSequence](
        self,
        *,
        nominal_input: ControlInputSequenceT,
        optimal_input: ControlInputSequenceT,
    ) -> ControlInputSequenceT:
        return optimal_input


class NoFilter:
    """Returns the optimal input unchanged."""

    def __call__[ControlInputSequenceT: ControlInputSequence](
        self, *, optimal_input: ControlInputSequenceT
    ) -> ControlInputSequenceT:
        return optimal_input
