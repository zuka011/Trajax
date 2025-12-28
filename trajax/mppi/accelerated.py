from typing import cast
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    JaxState,
    JaxStateBatch,
    JaxControlInputSequence,
    JaxControlInputBatch,
    JaxDynamicalModel,
    JaxCosts,
    JaxCostFunction,
    JaxSampler,
    JaxUpdateFunction,
    JaxPaddingFunction,
    JaxFilterFunction,
    Mppi,
    Control,
)
from trajax.mppi.common import UseOptimalControlUpdate, NoFilter

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


class JaxZeroPadding(
    JaxPaddingFunction[JaxControlInputSequence, JaxControlInputSequence]
):
    def __call__[T: int, D_u: int, L: int](
        self, *, nominal_input: JaxControlInputSequence[T, D_u], padding_size: L
    ) -> JaxControlInputSequence[L, D_u]:
        padding_array = jnp.zeros((padding_size, nominal_input.array.shape[1]))
        return nominal_input.similar(array=padding_array, length=padding_size)


@dataclass(kw_only=True, frozen=True)
class JaxMppi[
    StateT: JaxState,
    StateBatchT: JaxStateBatch,
    ControlInputSequenceT: JaxControlInputSequence,
    ControlInputBatchT: JaxControlInputBatch,
    ControlInputPaddingT: JaxControlInputSequence = ControlInputSequenceT,
](Mppi[StateT, ControlInputSequenceT]):
    planning_interval: int
    model: JaxDynamicalModel[
        StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ]
    cost_function: JaxCostFunction[ControlInputBatchT, StateBatchT, JaxCosts]
    sampler: JaxSampler[ControlInputSequenceT, ControlInputBatchT]
    update_function: JaxUpdateFunction[ControlInputSequenceT]
    padding_function: JaxPaddingFunction[ControlInputSequenceT, ControlInputPaddingT]
    filter_function: JaxFilterFunction[ControlInputSequenceT]

    @staticmethod
    def create[
        S: JaxState,
        SB: JaxStateBatch,
        CIS: JaxControlInputSequence,
        CIB: JaxControlInputBatch,
        CIP: JaxControlInputSequence = CIS,
    ](
        *,
        planning_interval: int = 1,
        model: JaxDynamicalModel[S, SB, CIS, CIB],
        cost_function: JaxCostFunction[CIB, SB, JaxCosts],
        sampler: JaxSampler[CIS, CIB],
        update_function: JaxUpdateFunction[CIS] | None = None,
        padding_function: JaxPaddingFunction[CIS, CIP] | None = None,
        filter_function: JaxFilterFunction[CIS] | None = None,
    ) -> "JaxMppi[S, SB, CIS, CIB, CIP]":
        """Creates a JAX-based MPPI controller."""
        return JaxMppi(
            planning_interval=planning_interval,
            model=model,
            cost_function=cost_function,
            sampler=sampler,
            update_function=update_function
            or cast(JaxUpdateFunction[CIS], UseOptimalControlUpdate()),
            padding_function=padding_function
            or cast(JaxPaddingFunction[CIS, CIP], JaxZeroPadding()),
            filter_function=filter_function or cast(JaxFilterFunction[CIS], NoFilter()),
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

        rollouts = self.model.simulate(inputs=samples, initial_state=initial_state)
        costs = self.cost_function(inputs=samples, states=rollouts)

        optimal_control = compute_weighted_control(
            samples.array, costs.array, temperature
        )
        optimal_input = self.filter_function(
            optimal_input=nominal_input.similar(array=optimal_control)
        )

        nominal_input = self.update_function(
            nominal_input=nominal_input, optimal_input=optimal_input
        )

        shifted_control = shift_control_left(
            nominal_input.array,
            padding_array=self.padding_function(
                nominal_input=nominal_input, padding_size=self.planning_interval
            ).array,
            planning_interval=self.planning_interval,
        )

        return Control(
            optimal=optimal_input, nominal=nominal_input.similar(array=shifted_control)
        )


@jax.jit
@jaxtyped
def compute_weighted_control(
    samples: Float[JaxArray, "T D_u M"],
    costs: Float[JaxArray, "T M"],
    temperature: Scalar,
) -> Float[JaxArray, "T D_u"]:
    total_costs = jnp.sum(costs, axis=0)
    min_cost = jnp.min(total_costs)
    exp_costs = jnp.exp((total_costs - min_cost) / (-temperature))
    normalizing_constant = jnp.sum(exp_costs)
    weights = exp_costs / normalizing_constant

    return jnp.tensordot(samples, weights, axes=([2], [0]))


@jax.jit(static_argnames=("planning_interval",))
@jaxtyped
def shift_control_left(
    control: Float[JaxArray, "T D_u"],
    *,
    padding_array: Float[JaxArray, "P D_u"],
    planning_interval: int,
) -> Float[JaxArray, "T D_u"]:
    return jnp.concatenate([control[planning_interval:], padding_array], axis=0)
