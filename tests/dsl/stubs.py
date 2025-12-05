from dataclasses import dataclass

from trajax import State, StateBatch, ControlInputBatch, ControlInputSequence

import numpy as np


@dataclass(frozen=True)
class Sampler[SequenceT: ControlInputSequence, BatchT: ControlInputBatch]:
    samples: BatchT
    expected_nominal_input: SequenceT

    @staticmethod
    def returns[S: ControlInputSequence, B: ControlInputBatch](
        sampled_inputs: B, *, when_nominal_input_is: S
    ) -> "Sampler[S, B]":
        return Sampler(
            samples=sampled_inputs, expected_nominal_input=when_nominal_input_is
        )

    def sample(self, *, around: SequenceT) -> BatchT:
        assert np.array_equal(self.expected_nominal_input, around), (
            f"Sampler received an unexpected nominal input. "
            f"Expected: {self.expected_nominal_input}, Got: {around}"
        )
        return self.samples


@dataclass(frozen=True)
class DynamicalModel[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
]:
    rollouts: StateBatchT
    expected_control_inputs: ControlInputBatchT
    expected_initial_state: StateT

    @staticmethod
    def returns[SB: StateBatch, CIB: ControlInputBatch, S: State](
        rollouts: SB, *, when_control_inputs_are: CIB, and_initial_state_is: S
    ) -> "DynamicalModel[CIB, S, SB]":
        return DynamicalModel(
            rollouts=rollouts,
            expected_control_inputs=when_control_inputs_are,
            expected_initial_state=and_initial_state_is,
        )

    async def simulate(
        self, inputs: ControlInputBatchT, initial_state: StateT
    ) -> StateBatchT:
        assert np.array_equal(self.expected_control_inputs, inputs), (
            f"Dynamical model received unexpected control inputs. "
            f"Expected: {self.expected_control_inputs}, Got: {inputs}"
        )
        assert np.array_equal(self.expected_initial_state, initial_state), (
            f"Dynamical model received an unexpected initial state. "
            f"Expected: {self.expected_initial_state}, Got: {initial_state}"
        )
        return self.rollouts

    async def step(self, input: ControlInputSequence, state: StateT) -> StateT:
        raise NotImplementedError("Step method is not implemented in the stub model.")
