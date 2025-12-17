from typing import Protocol, Self, overload, cast
from dataclasses import dataclass

from trajax.type import jaxtyped
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

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


class JaxState[D_x: int](State[D_x], Protocol):
    @property
    def array(self) -> Float[JaxArray, "D_x"]:
        """Returns the underlying JAX array representing the state."""
        ...


class JaxStateBatch[T: int, D_x: int, M: int](StateBatch[T, D_x, M], Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_x M"]:
        """Returns the underlying JAX array representing the state batch."""
        ...


class JaxControlInputSequence[T: int, D_u: int](ControlInputSequence[T, D_u], Protocol):
    @overload
    def similar(self, *, array: Float[JaxArray, "T D_u"]) -> Self:
        """Creates a new control input sequence similar to this one but with the given
        array as its data.
        """
        ...

    @overload
    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L
    ) -> "JaxControlInputSequence[L, D_u]":
        """Returns a new control input sequence similar to this one but using the provided array.
        The length of the new sequence may differ from the original."""
        ...

    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        """Returns the underlying JAX array representing the control input sequence."""
        ...


class JaxControlInputBatch[T: int, D_u: int, M: int](
    ControlInputBatch[T, D_u, M], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_u M"]:
        """Returns the underlying JAX array representing the control input batch."""
        ...


class JaxCosts[T: int, M: int](Costs[T, M], Protocol):
    @property
    def array(self) -> Float[JaxArray, "T M"]:
        """Returns the underlying JAX array representing the costs."""
        ...


class JaxDynamicalModel[
    InStateT: JaxState,
    OutStateT: JaxState,
    StateBatchT: JaxStateBatch,
    ControlInputSequenceT: JaxControlInputSequence,
    ControlInputBatchT: JaxControlInputBatch,
](
    DynamicalModel[
        InStateT, OutStateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
    ],
    Protocol,
): ...


class JaxSampler[SequenceT: JaxControlInputSequence, BatchT: JaxControlInputBatch](
    Sampler[SequenceT, BatchT], Protocol
): ...


class JaxCostFunction[
    ControlInputBatchT: JaxControlInputBatch,
    StateBatchT: JaxStateBatch,
    CostsT: JaxCosts,
](CostFunction[ControlInputBatchT, StateBatchT, CostsT], Protocol): ...


class JaxUpdateFunction[C: JaxControlInputSequence](UpdateFunction[C], Protocol): ...


class JaxPaddingFunction[N: JaxControlInputSequence, P: JaxControlInputSequence](
    PaddingFunction[N, P], Protocol
): ...


class JaxFilterFunction[C: JaxControlInputSequence](FilterFunction[C], Protocol): ...


class JaxZeroPadding[T: int, D_u: int, L: int]:
    def __call__(
        self, *, nominal_input: JaxControlInputSequence[T, D_u], padding_size: L
    ) -> JaxControlInputSequence[L, D_u]:
        padding_array = jnp.zeros((padding_size, nominal_input.array.shape[1]))
        return nominal_input.similar(array=padding_array, length=padding_size)


@dataclass(kw_only=True, frozen=True)
class JaxMppi[
    ControlInputSequenceT: JaxControlInputSequence,
    ControlInputPaddingT: JaxControlInputSequence,
](
    Mppi[
        JaxState,
        JaxState,
        JaxStateBatch,
        ControlInputSequenceT,
        JaxControlInputBatch,
        JaxCosts,
    ]
):
    planning_interval: int
    update_function: JaxUpdateFunction[ControlInputSequenceT]
    padding_function: JaxPaddingFunction[ControlInputSequenceT, ControlInputPaddingT]
    filter_function: JaxFilterFunction[ControlInputSequenceT]

    @staticmethod
    def create[
        CIS: JaxControlInputSequence = JaxControlInputSequence,
        CIP: JaxControlInputSequence = JaxControlInputSequence,
    ](
        *,
        planning_interval: int = 1,
        update_function: JaxUpdateFunction[CIS] | None = None,
        padding_function: JaxPaddingFunction[CIS, CIP] | None = None,
        filter_function: JaxFilterFunction[CIS] | None = None,
    ) -> "JaxMppi[CIS, CIP]":
        """Creates a JAX-based MPPI controller."""
        return JaxMppi(
            planning_interval=planning_interval,
            update_function=update_function
            or cast(JaxUpdateFunction[CIS], UseOptimalControlUpdate()),
            padding_function=padding_function
            or cast(JaxPaddingFunction[CIS, CIP], JaxZeroPadding()),
            filter_function=filter_function or cast(JaxFilterFunction[CIS], NoFilter()),
        )

    def __post_init__(self) -> None:
        assert self.planning_interval > 0, "Planning interval must be positive."

    async def step[
        InStateT: JaxState,
        OutStateT: JaxState,
        StateBatchT: JaxStateBatch,
        ControlInputBatchT: JaxControlInputBatch,
        CostsT: JaxCosts,
    ](
        self,
        *,
        model: DynamicalModel[
            InStateT, OutStateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
        ],
        cost_function: JaxCostFunction[ControlInputBatchT, StateBatchT, CostsT],
        sampler: Sampler[ControlInputSequenceT, ControlInputBatchT],
        temperature: float,
        nominal_input: ControlInputSequenceT,
        initial_state: InStateT,
    ) -> Control[ControlInputSequenceT]:
        assert temperature > 0.0, "Temperature must be positive."

        samples = sampler.sample(around=nominal_input)

        rollouts = await model.simulate(inputs=samples, initial_state=initial_state)
        costs = cost_function(inputs=samples, states=rollouts)

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
