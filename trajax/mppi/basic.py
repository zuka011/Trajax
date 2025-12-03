from typing import Protocol

from trajax.model import (
    DynamicalModel as AnyDynamicalModel,
    State,
    StateBatch,
    ControlInputBatch,
)
from trajax.mppi.common import (
    Control,
    Sampler as AnySampler,
    CostFunction as AnyCostFunction,
    ControlInputSequence as AnyControlInputSequence,
)

import numpy as np
from numtypes import Array, Dims, shape_of


class NumPyMppi:
    class ControlInputSequence[T: int, D_u: int](
        AnyControlInputSequence[T, D_u], Protocol
    ):
        def similar(
            self, *, array: Array[Dims[T, D_u]]
        ) -> "NumPyMppi.ControlInputSequence[T, D_u]":
            """Creates a new control input sequence similar to this one but with the given
            array as its data.
            """
            ...

    type DynamicalModel[T: int, D_u: int, D_x: int, M: int] = AnyDynamicalModel[
        ControlInputBatch[T, D_u, M],
        State[D_x],
        StateBatch[T, D_x, M],
    ]

    type CostFunction[T: int, D_u: int, D_x: int, M: int] = AnyCostFunction[
        ControlInputBatch[T, D_u, M],
        StateBatch[T, D_x, M],
        Array[Dims[T, M]],
    ]

    type Sampler[T: int, D_u: int, M: int] = AnySampler[
        NumPyMppi.ControlInputSequence[T, D_u], ControlInputBatch[T, D_u, M]
    ]

    @staticmethod
    def create() -> "NumPyMppi":
        return NumPyMppi()

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

        temperature = 1.0  # TODO: Make sure temperature is considered.
        min_cost = np.min(costs_per_rollout)
        exp_costs = np.exp((costs_per_rollout - min_cost) / (-temperature))

        # normalizing_constant = exp_costs.sum() # TODO: Make sure weights are normalized.
        weights = exp_costs  # / normalizing_constant

        optimal_control = np.tensordot(samples, weights, axes=([2], [0]))

        return Control(optimal=nominal_input.similar(array=optimal_control))
