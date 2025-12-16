from typing import Protocol, Self, overload
from dataclasses import dataclass

from trajax.model import (
    DynamicalModel,
    State,
    StateBatch,
    ControlInputSequence as AnyControlInputSequence,
    ControlInputBatch,
)
from trajax.mppi.common import (
    Mppi,
    Control,
    UseOptimalControlUpdate,
    NoFilter,
    UpdateFunction as AnyUpdateFunction,
    PaddingFunction as AnyPaddingFunction,
    FilterFunction as AnyFilterFunction,
    Sampler,
    CostFunction,
    Costs as AnyCosts,
)

from numtypes import Array, Dims, shape_of

import numpy as np


class ControlInputSequence[T: int = int, D_u: int = int](
    AnyControlInputSequence[T, D_u], Protocol
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
    ) -> "ControlInputSequence[L, D_u]":
        """Creates a new control input sequence similar to this one but with the given
        array as its data. The length of the new sequence may differ from the original.
        """
        ...


class Costs[T: int = int, M: int = int](AnyCosts[T, M], Protocol):
    def similar(self, *, array: Array[Dims[T, M]]) -> Self:
        """Creates new costs similar to this one but with the given array as its data."""
        ...


type UpdateFunction[C: ControlInputSequence] = AnyUpdateFunction[C]

type PaddingFunction[N: ControlInputSequence, P: ControlInputSequence] = (
    AnyPaddingFunction[N, P]
)

type FilterFunction[C: ControlInputSequence] = AnyFilterFunction[C]


class ZeroPadding[T: int = int, D_u: int = int, L: int = int]:
    def __call__(
        self, *, nominal_input: ControlInputSequence[T, D_u], padding_size: L
    ) -> ControlInputSequence[L, D_u]:
        array = np.zeros(shape := (padding_size, nominal_input.dimension))

        assert shape_of(array, matches=shape, name="padding array")

        return nominal_input.similar(array=array, length=padding_size)


@dataclass(kw_only=True, frozen=True)
class NumPyMppi[
    InStateT: State,
    OutStateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputPaddingT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
    CostsT: Costs,
](
    Mppi[
        InStateT,
        OutStateT,
        StateBatchT,
        ControlInputSequenceT,
        ControlInputBatchT,
        CostsT,
    ]
):
    planning_interval: int
    update_function: UpdateFunction[ControlInputSequenceT]
    padding_function: PaddingFunction[ControlInputSequenceT, ControlInputPaddingT]
    filter_function: FilterFunction[ControlInputSequenceT]

    @staticmethod
    def create[
        IS: State = State,
        OS: State = State,
        SB: StateBatch = StateBatch,
        CIS: ControlInputSequence = ControlInputSequence,
        CIP: ControlInputSequence = ControlInputSequence,
        CIB: ControlInputBatch = ControlInputBatch,
        C: Costs = Costs,
    ](
        *,
        planning_interval: int = 1,
        update_function: UpdateFunction[CIS] = UseOptimalControlUpdate(),
        padding_function: PaddingFunction[CIS, CIP] = ZeroPadding(),
        filter_function: FilterFunction[CIS] = NoFilter(),
    ) -> "NumPyMppi[IS, OS, SB, CIS, CIP, CIB, C]":
        """Creates a NumPy-based MPPI controller."""
        return NumPyMppi(
            planning_interval=planning_interval,
            update_function=update_function,
            padding_function=padding_function,
            filter_function=filter_function,
        )

    def __post_init__(self) -> None:
        assert self.planning_interval > 0, "Planning interval must be positive."

    async def step(
        self,
        *,
        model: DynamicalModel[
            InStateT, OutStateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
        ],
        cost_function: CostFunction[ControlInputBatchT, StateBatchT, Costs],
        sampler: Sampler[ControlInputSequenceT, ControlInputBatchT],
        temperature: float,
        nominal_input: ControlInputSequenceT,
        initial_state: InStateT,
    ) -> Control[ControlInputSequenceT]:
        assert temperature > 0.0, "Temperature must be positive."

        samples = sampler.sample(around=nominal_input)
        rollout_count = samples.rollout_count

        rollouts = await model.simulate(inputs=samples, initial_state=initial_state)
        costs = cost_function(inputs=samples, states=rollouts)
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
