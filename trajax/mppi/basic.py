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
    NoUpdate,
    UpdateFunction as AnyUpdateFunction,
    Sampler as AnySampler,
    CostFunction as AnyCostFunction,
)

import numpy as np
from numtypes import Array, Dims, shape_of


class NumPyZeroPadding:
    pass


class ControlInputSequence[T: int, D_u: int](AnyControlInputSequence[T, D_u], Protocol):
    def similar(self, *, array: Array[Dims[T, D_u]]) -> "ControlInputSequence[T, D_u]":
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


@dataclass(kw_only=True, frozen=True)
class NumPyMppi:
    type CostFunction[T: int, D_u: int, D_x: int, M: int] = AnyCostFunction[
        ControlInputBatch[T, D_u, M], StateBatch[T, D_x, M], Array[Dims[T, M]]
    ]

    planning_interval: int
    update_function: UpdateFunction

    @staticmethod
    def create(
        *,
        planning_interval: int = 1,
        update_function: UpdateFunction = NoUpdate(),
        padding_function: ... = None,
    ) -> "NumPyMppi":
        """Creates a NumPy-based MPPI controller."""
        return NumPyMppi(
            planning_interval=planning_interval, update_function=update_function
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
        control_dimension = optimal_control.shape[1]

        optimal_input = nominal_input.similar(array=optimal_control)
        nominal_input = self.update_function(
            nominal_input=nominal_input, optimal_input=optimal_input
        )

        shifted_control = np.concat(
            [
                np.asarray(nominal_input)[self.planning_interval :],
                np.zeros((self.planning_interval, control_dimension)),
            ],
            axis=0,
        )

        return Control(
            optimal=optimal_input, nominal=nominal_input.similar(array=shifted_control)
        )
