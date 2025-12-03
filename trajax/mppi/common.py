from typing import Protocol
from dataclasses import dataclass

from trajax.type import DataType
from trajax.model import DynamicalModel, State, StateBatch, ControlInputBatch

from numtypes import Array, Dims


class ControlInputSequence[T: int = int, D_u: int = int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        """Returns the control input sequence as a NumPy array."""
        ...


class Costs(Protocol): ...


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
        model: DynamicalModel[ControlInputBatchT, StateT, StateBatchT],
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
