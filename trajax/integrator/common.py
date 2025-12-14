from typing import Protocol

from trajax.model import ControlInputBatch, State, StateBatch


class IntegratorModel[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](Protocol):
    async def simulate(
        self, inputs: ControlInputBatchT, initial_state: StateT
    ) -> StateBatchT:
        """Simulates a single integrator model over the given control inputs starting from the
        initial state."""
        ...
