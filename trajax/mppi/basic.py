from typing import cast, Any
from dataclasses import dataclass

from trajax.types import (
    DataType,
    NumPyState,
    NumPyStateBatch,
    NumPyControlInputSequence,
    NumPyControlInputBatch,
    NumPyDynamicalModel,
    NumPyCosts,
    NumPyCostFunction,
    NumPySampler,
    NumPyUpdateFunction,
    NumPyPaddingFunction,
    NumPyFilterFunction,
    Mppi,
    DebugData,
    Control,
)
from trajax.mppi.common import UseOptimalControlUpdate, NoFilter

from numtypes import Array, Dim2, Dims, shape_of

import numpy as np


@dataclass(frozen=True)
class NumPyWeights[M: int]:
    _array: Array[Dims[M]]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[M]]:
        return np.array(self._array, dtype=dtype)

    @property
    def rollout_count(self) -> M:
        return self._array.shape[0]

    @property
    def array(self) -> Array[Dims[M]]:
        return self._array


class NumPyZeroPadding(
    NumPyPaddingFunction[NumPyControlInputSequence, NumPyControlInputSequence]
):
    def __call__[T: int, D_u: int, L: int](
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
](Mppi[StateT, ControlInputSequenceT, NumPyWeights[int]]):
    planning_interval: int
    model: NumPyDynamicalModel[
        StateT, Any, StateBatchT, ControlInputSequenceT, ControlInputBatchT
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
        model: NumPyDynamicalModel[S, Any, SB, CIS, CIB],
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
    ) -> Control[ControlInputSequenceT, NumPyWeights]:
        assert temperature > 0.0, "Temperature must be positive."

        samples = self.sampler.sample(around=nominal_input)
        rollout_count = samples.rollout_count

        rollouts = self.model.simulate(inputs=samples, initial_state=initial_state)
        costs = self.cost_function(inputs=samples, states=rollouts)
        costs_per_rollout = np.sum(costs.array, axis=0)

        assert shape_of(
            costs_per_rollout, matches=(rollout_count,), name="costs per rollout"
        )

        min_cost = np.min(costs_per_rollout)
        exp_costs = np.exp((costs_per_rollout - min_cost) / (-temperature))

        normalizing_constant = exp_costs.sum()
        weights = exp_costs / normalizing_constant

        optimal_control = cast(
            Array[Dim2], np.tensordot(samples, weights, axes=([2], [0]))
        )
        optimal_input = self.filter_function(
            optimal_input=nominal_input.similar(array=optimal_control)
        )

        nominal_input = self.update_function(
            nominal_input=nominal_input, optimal_input=optimal_input
        )

        shifted_control = np.concat(
            [
                nominal_input.array[self.planning_interval :],
                self.padding_function(
                    nominal_input=nominal_input, padding_size=self.planning_interval
                ).array,
            ],
            axis=0,
        )

        return Control(
            optimal=optimal_input,
            nominal=nominal_input.similar(array=shifted_control),
            debug=DebugData(trajectory_weights=NumPyWeights(weights)),
        )
