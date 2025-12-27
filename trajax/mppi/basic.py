from typing import Protocol, Self, cast, overload
from dataclasses import dataclass

from trajax.mppi.common import (
    State,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
    DynamicalModel,
    Costs,
    CostFunction,
    Sampler,
    Mppi,
    Control,
    UseOptimalControlUpdate,
    NoFilter,
    UpdateFunction,
    PaddingFunction,
    FilterFunction,
)

from numtypes import Array, Dims, shape_of

import numpy as np


class NumPyState[D_x: int](State[D_x], Protocol): ...


class NumPyStateBatch[T: int, D_x: int, M: int](StateBatch[T, D_x, M], Protocol): ...


class NumPyControlInputSequence[T: int, D_u: int](
    ControlInputSequence[T, D_u], Protocol
):
    @overload
    def similar(self, *, array: Array[Dims[T, D_u]]) -> Self:
        """Creates a new control input sequence similar to this one but with the given
        array as its data.
        """
        ...

    @overload
    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]], length: L
    ) -> "NumPyControlInputSequence[L, D_u]":
        """Creates a new control input sequence similar to this one but with the given
        array as its data. The length of the new sequence may differ from the original.
        """
        ...


class NumPyControlInputBatch[T: int, D_u: int, M: int](
    ControlInputBatch[T, D_u, M], Protocol
): ...


class NumPyCosts[T: int, M: int](Costs[T, M], Protocol):
    def similar(self, *, array: Array[Dims[T, M]]) -> Self:
        """Creates new costs similar to this one but with the given array as its data."""
        ...


class NumPyDynamicalModel[
    StateT: NumPyState,
    StateBatchT: NumPyStateBatch,
    ControlInputSequenceT: NumPyControlInputSequence,
    ControlInputBatchT: NumPyControlInputBatch,
](
    DynamicalModel[StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT],
    Protocol,
): ...


class NumPySampler[
    SequenceT: NumPyControlInputSequence,
    BatchT: NumPyControlInputBatch,
](Sampler[SequenceT, BatchT], Protocol): ...


class NumPyCostFunction[
    ControlInputBatchT: NumPyControlInputBatch,
    StateBatchT: NumPyStateBatch,
    CostsT: NumPyCosts,
](CostFunction[ControlInputBatchT, StateBatchT, CostsT], Protocol): ...


class NumPyUpdateFunction[C: NumPyControlInputSequence](
    UpdateFunction[C], Protocol
): ...


class NumPyPaddingFunction[
    N: NumPyControlInputSequence,
    P: NumPyControlInputSequence,
](PaddingFunction[N, P], Protocol): ...


class NumPyFilterFunction[C: NumPyControlInputSequence](
    FilterFunction[C], Protocol
): ...


class NumPyZeroPadding[T: int, D_u: int, L: int]:
    def __call__(
        self, *, nominal_input: NumPyControlInputSequence[T, D_u], padding_size: L
    ) -> NumPyControlInputSequence[L, D_u]:
        array = np.zeros(shape := (padding_size, nominal_input.dimension))

        assert shape_of(array, matches=shape, name="padding array")

        return nominal_input.similar(array=array, length=padding_size)


@dataclass(kw_only=True, frozen=True)
class NumPyMppi[
    StateT: NumPyState,
    StateBatchT: NumPyStateBatch,
    ControlInputSequenceT: NumPyControlInputSequence,
    ControlInputBatchT: NumPyControlInputBatch,
    ControlInputPaddingT: NumPyControlInputSequence = ControlInputSequenceT,
](Mppi[StateT, ControlInputSequenceT]):
    planning_interval: int
    model: NumPyDynamicalModel[
        StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ]
    cost_function: NumPyCostFunction[ControlInputBatchT, StateBatchT, NumPyCosts]
    sampler: NumPySampler[ControlInputSequenceT, ControlInputBatchT]
    update_function: NumPyUpdateFunction[ControlInputSequenceT]
    padding_function: NumPyPaddingFunction[ControlInputSequenceT, ControlInputPaddingT]
    filter_function: NumPyFilterFunction[ControlInputSequenceT]

    @staticmethod
    def create[
        S: NumPyState,
        SB: NumPyStateBatch,
        CIS: NumPyControlInputSequence,
        CIB: NumPyControlInputBatch,
        CIP: NumPyControlInputSequence = CIS,
    ](
        *,
        planning_interval: int = 1,
        model: NumPyDynamicalModel[S, SB, CIS, CIB],
        cost_function: NumPyCostFunction[CIB, SB, NumPyCosts],
        sampler: NumPySampler[CIS, CIB],
        update_function: NumPyUpdateFunction[CIS] | None = None,
        padding_function: NumPyPaddingFunction[CIS, CIP] | None = None,
        filter_function: NumPyFilterFunction[CIS] | None = None,
    ) -> "NumPyMppi[S, SB, CIS, CIB, CIP]":
        """Creates a NumPy-based MPPI controller."""
        return NumPyMppi(
            planning_interval=planning_interval,
            model=model,
            cost_function=cost_function,
            sampler=sampler,
            update_function=update_function
            or cast(NumPyUpdateFunction[CIS], UseOptimalControlUpdate()),
            padding_function=padding_function
            or cast(NumPyPaddingFunction[CIS, CIP], NumPyZeroPadding()),
            filter_function=filter_function
            or cast(NumPyFilterFunction[CIS], NoFilter()),
        )

    def __post_init__(self) -> None:
        assert self.planning_interval > 0, "Planning interval must be positive."

    def step(
        self,
        *,
        temperature: float,
        nominal_input: ControlInputSequenceT,
        initial_state: StateT,
    ) -> Control[ControlInputSequenceT]:
        assert temperature > 0.0, "Temperature must be positive."

        samples = self.sampler.sample(around=nominal_input)
        rollout_count = samples.rollout_count

        rollouts = self.model.simulate(inputs=samples, initial_state=initial_state)
        costs = self.cost_function(inputs=samples, states=rollouts)
        costs_per_rollout = np.sum(costs, axis=0)

        assert shape_of(
            costs_per_rollout, matches=(rollout_count,), name="costs per rollout"
        )

        min_cost = np.min(costs_per_rollout)
        exp_costs = np.exp((costs_per_rollout - min_cost) / (-temperature))

        normalizing_constant = exp_costs.sum()
        weights = exp_costs / normalizing_constant

        optimal_control = np.tensordot(samples, weights, axes=([2], [0]))
        optimal_input = self.filter_function(
            optimal_input=nominal_input.similar(array=optimal_control)
        )

        nominal_input = self.update_function(
            nominal_input=nominal_input, optimal_input=optimal_input
        )

        shifted_control = np.concat(
            [
                np.asarray(nominal_input)[self.planning_interval :],
                np.asarray(
                    self.padding_function(
                        nominal_input=nominal_input, padding_size=self.planning_interval
                    )
                ),
            ],
            axis=0,
        )

        return Control(
            optimal=optimal_input, nominal=nominal_input.similar(array=shifted_control)
        )
