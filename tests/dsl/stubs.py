from dataclasses import dataclass

from trajax import (
    State,
    StateBatch,
    ControlInputBatch,
    ControlInputSequence,
    Sampler as SamplerLike,
    UpdateFunction as UpdateFunctionLike,
    PaddingFunction as PaddingFunctionLike,
    FilterFunction as FilterFunctionLike,
    DynamicalModel as DynamicalModelLike,
)

import numpy as np


@dataclass(frozen=True)
class Sampler[SequenceT: ControlInputSequence, BatchT: ControlInputBatch](
    SamplerLike[SequenceT, BatchT]
):
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

    @property
    def rollout_count(self) -> int:
        return self.samples.rollout_count


@dataclass(frozen=True)
class UpdateFunction[SequenceT: ControlInputSequence](UpdateFunctionLike[SequenceT]):
    expected_nominal_input: SequenceT
    expected_optimal_input: SequenceT
    result: SequenceT

    @staticmethod
    def returns[S: ControlInputSequence](
        result: S,
        *,
        when_nominal_input_is: S,
        and_optimal_input_is: S,
    ) -> "UpdateFunction[S]":
        return UpdateFunction(
            expected_nominal_input=when_nominal_input_is,
            expected_optimal_input=and_optimal_input_is,
            result=result,
        )

    def __call__(
        self, *, nominal_input: SequenceT, optimal_input: SequenceT
    ) -> SequenceT:
        assert np.array_equal(self.expected_nominal_input, nominal_input), (
            f"UpdateFunction received an unexpected nominal input. "
            f"Expected: {self.expected_nominal_input}, Got: {nominal_input}"
        )
        assert np.allclose(self.expected_optimal_input, optimal_input, atol=1e-6), (
            f"UpdateFunction received an unexpected optimal input. "
            f"Expected: {self.expected_optimal_input}, Got: {optimal_input}"
        )
        return self.result


@dataclass(frozen=True)
class PaddingFunction[NominalT: ControlInputSequence, PaddingT: ControlInputSequence](
    PaddingFunctionLike[NominalT, PaddingT]
):
    expected_nominal_input: NominalT
    expected_padding_size: int
    result: PaddingT

    @staticmethod
    def returns[N: ControlInputSequence, P: ControlInputSequence](
        result: P,
        *,
        when_nominal_input_is: N,
        and_padding_size_is: int,
    ) -> "PaddingFunction[N, P]":
        return PaddingFunction(
            expected_nominal_input=when_nominal_input_is,
            expected_padding_size=and_padding_size_is,
            result=result,
        )

    def __call__(self, *, nominal_input: NominalT, padding_size: int) -> PaddingT:
        assert np.array_equal(self.expected_nominal_input, nominal_input), (
            f"PaddingFunction received an unexpected nominal input. "
            f"Expected: {self.expected_nominal_input}, Got: {nominal_input}"
        )
        assert self.expected_padding_size == padding_size, (
            f"PaddingFunction received an unexpected padding size. "
            f"Expected: {self.expected_padding_size}, Got: {padding_size}"
        )
        return self.result


@dataclass(frozen=True)
class FilterFunction[SequenceT: ControlInputSequence](FilterFunctionLike[SequenceT]):
    expected_optimal_input: SequenceT
    result: SequenceT

    @staticmethod
    def returns[S: ControlInputSequence](
        result: S,
        *,
        when_optimal_input_is: S,
    ) -> "FilterFunction[S]":
        return FilterFunction(
            expected_optimal_input=when_optimal_input_is,
            result=result,
        )

    def __call__(self, *, optimal_input: SequenceT) -> SequenceT:
        assert np.allclose(self.expected_optimal_input, optimal_input, atol=1e-6), (
            f"FilterFunction received an unexpected optimal input. "
            f"Expected: {self.expected_optimal_input}, Got: {optimal_input}"
        )
        return self.result


@dataclass(frozen=True)
class DynamicalModel[
    InStateT: State,
    OutStateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
](
    DynamicalModelLike[
        InStateT, OutStateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ]
):
    rollouts: StateBatchT
    expected_control_inputs: ControlInputBatchT
    expected_initial_state: InStateT

    @staticmethod
    def returns[
        S: State,
        SB: StateBatch,
        CIB: ControlInputBatch,
    ](
        rollouts: SB, *, when_control_inputs_are: CIB, and_initial_state_is: S
    ) -> "DynamicalModel[S, State, SB, ControlInputSequence, CIB]":
        return DynamicalModel(
            rollouts=rollouts,
            expected_control_inputs=when_control_inputs_are,
            expected_initial_state=and_initial_state_is,
        )

    async def simulate(
        self, inputs: ControlInputBatchT, initial_state: InStateT
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

    async def step(self, input: ControlInputSequenceT, state: InStateT) -> OutStateT:
        raise NotImplementedError("Step method is not implemented in the stub model.")
