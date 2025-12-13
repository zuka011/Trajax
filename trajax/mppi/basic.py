from typing import Protocol
from dataclasses import dataclass

from trajax.model import (
    DynamicalModel as AnyDynamicalModel,
    State,
    StateBatch,
    ControlInputSequence as AnyControlInputSequence,
    ControlInputBatch,
)
from trajax.mppi.common import (
    Control,
    UseOptimalControlUpdate,
    NoFilter,
    UpdateFunction as AnyUpdateFunction,
    PaddingFunction as AnyPaddingFunction,
    FilterFunction as AnyFilterFunction,
    Sampler as AnySampler,
    CostFunction as AnyCostFunction,
)

import numpy as np
from numtypes import Array, Dims, shape_of


class ControlInputSequence[T: int, D_u: int](AnyControlInputSequence[T, D_u], Protocol):
    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]]
    ) -> "ControlInputSequence[L, D_u]":
        """Creates a new control input sequence similar to this one but with the given
        array as its data.
        """
        ...


type DynamicalModel[T: int, D_u: int, D_x: int, M: int] = AnyDynamicalModel[
    ControlInputSequence[T, D_u],
    ControlInputBatch[T, D_u, M],
    State[D_x],
    StateBatch[T, D_x, M],
]

type Sampler[T: int, D_u: int, M: int] = AnySampler[
    ControlInputSequence[T, D_u], ControlInputBatch[T, D_u, M]
]

type UpdateFunction[T: int, D_u: int] = AnyUpdateFunction[ControlInputSequence[T, D_u]]

type PaddingFunction[T: int, P: int, D_u: int] = AnyPaddingFunction[
    ControlInputSequence[T, D_u], ControlInputSequence[P, D_u]
]

type FilterFunction[T: int, D_u: int] = AnyFilterFunction[ControlInputSequence[T, D_u]]


@dataclass(kw_only=True, frozen=True)
class NumPyMppi:
    type CostFunction[T: int, D_u: int, D_x: int, M: int] = AnyCostFunction[
        ControlInputBatch[T, D_u, M], StateBatch[T, D_x, M], Array[Dims[T, M]]
    ]

    class ZeroPadding:
        def __call__[T: int, P: int, D_u: int](
            self, *, nominal_input: ControlInputSequence[T, D_u], padding_size: P
        ) -> ControlInputSequence[P, D_u]:
            array = np.zeros(shape := (padding_size, nominal_input.dimension))

            assert shape_of(array, matches=shape, name="padding array")

            return nominal_input.similar(array=array)

    planning_interval: int
    update_function: UpdateFunction
    padding_function: PaddingFunction
    filter_function: FilterFunction

    @staticmethod
    def create(
        *,
        planning_interval: int = 1,
        update_function: UpdateFunction = UseOptimalControlUpdate(),
        padding_function: PaddingFunction = ZeroPadding(),
        filter_function: FilterFunction = NoFilter(),
    ) -> "NumPyMppi":
        """Creates a NumPy-based MPPI controller."""
        return NumPyMppi(
            planning_interval=planning_interval,
            update_function=update_function,
            padding_function=padding_function,
            filter_function=filter_function,
        )

    def __post_init__(self) -> None:
        assert self.planning_interval > 0, "Planning interval must be positive."

    async def step[T: int, D_u: int, D_x: int, M: int](
        self,
        *,
        model: DynamicalModel[T, D_u, D_x, M],
        cost_function: CostFunction[T, D_u, D_x, M],
        sampler: Sampler[T, D_u, M],
        temperature: float,
        nominal_input: ControlInputSequence[T, D_u],
        initial_state: State[D_x],
    ) -> Control[ControlInputSequence[T, D_u]]:
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
