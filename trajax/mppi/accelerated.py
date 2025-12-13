from typing import Protocol
from dataclasses import dataclass

from trajax.type import jaxtyped
from trajax.model import (
    ControlInputSequence as AnyControlInputSequence,
    ControlInputBatch as AnyControlInputBatch,
    State as AnyState,
    StateBatch as AnyStateBatch,
    DynamicalModel as AnyDynamicalModel,
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
    Costs as AnyCosts,
)

from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp


class ControlInputSequence[T: int, D_u: int](AnyControlInputSequence[T, D_u], Protocol):
    def similar[L: int](
        self, *, array: Float[Array, "L D_u"], length: L
    ) -> "ControlInputSequence[L, D_u]":
        """Returns a new control input sequence similar to this one but using the provided array."""
        ...

    @property
    def array(self) -> Float[Array, "T D_u"]:
        """Returns the underlying JAX array representing the control input sequence."""
        ...


class State[D_x: int](AnyState[D_x], Protocol):
    @property
    def array(self) -> Float[Array, "D_x"]:
        """Returns the underlying JAX array representing the state."""
        ...


type DynamicalModel[T: int, D_u: int, D_x: int, M: int] = AnyDynamicalModel[
    ControlInputSequence[T, D_u],
    "JaxMppi.ControlInputBatch[T, D_u, M]",
    State[D_x],
    "JaxMppi.StateBatch[T, D_x, M]",
]

type Sampler[T: int, D_u: int, M: int] = AnySampler[
    ControlInputSequence[T, D_u], "JaxMppi.ControlInputBatch[T, D_u, M]"
]

type UpdateFunction[T: int, D_u: int] = AnyUpdateFunction[ControlInputSequence[T, D_u]]

type PaddingFunction[T: int, P: int, D_u: int] = AnyPaddingFunction[
    ControlInputSequence[T, D_u], ControlInputSequence[P, D_u]
]

type FilterFunction[T: int, D_u: int] = AnyFilterFunction[ControlInputSequence[T, D_u]]


@dataclass(kw_only=True, frozen=True)
class JaxMppi:
    class ControlInputBatch[T: int, D_u: int, M: int](
        AnyControlInputBatch[T, D_u, M], Protocol
    ):
        @property
        def array(self) -> Float[Array, "T D_u M"]:
            """Returns the underlying JAX array representing the control input batch."""
            ...

    class StateBatch[T: int, D_x: int, M: int](AnyStateBatch[T, D_x, M], Protocol):
        @property
        def array(self) -> Float[Array, "T D_x M"]:
            """Returns the underlying JAX array representing the state batch."""
            ...

    class Costs[T: int, M: int](AnyCosts[T, M], Protocol):
        @property
        def array(self) -> Float[Array, "T M"]:
            """Returns the underlying JAX array representing the costs."""
            ...

    type CostFunction[T: int, D_u: int, D_x: int, M: int] = AnyCostFunction[
        "JaxMppi.ControlInputBatch[T, D_u, M]",
        "JaxMppi.StateBatch[T, D_x, M]",
        "JaxMppi.Costs[T, M]",
    ]

    class ZeroPadding:
        def __call__[T: int, P: int, D_u: int](
            self, *, nominal_input: ControlInputSequence[T, D_u], padding_size: P
        ) -> ControlInputSequence[P, D_u]:
            padding_array = jnp.zeros((padding_size, nominal_input.array.shape[1]))
            return nominal_input.similar(array=padding_array, length=padding_size)

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
    ) -> "JaxMppi":
        """Creates a JAX-based MPPI controller."""
        return JaxMppi(
            planning_interval=planning_interval,
            update_function=update_function,
            padding_function=padding_function,
            filter_function=filter_function,
        )

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

        rollouts = await model.simulate(inputs=samples, initial_state=initial_state)
        costs = cost_function(inputs=samples, states=rollouts)

        optimal_control = compute_weighted_control(
            samples.array, costs.array, temperature
        )
        optimal_input = self.filter_function(
            optimal_input=nominal_input.similar(
                array=optimal_control, length=optimal_control.shape[0]
            )
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
            optimal=optimal_input,
            nominal=nominal_input.similar(
                array=shifted_control, length=shifted_control.shape[0]
            ),
        )


@jax.jit
@jaxtyped
def compute_weighted_control(
    samples: Float[Array, "T D_u M"], costs: Float[Array, "T M"], temperature: Scalar
) -> Float[Array, "T D_u"]:
    total_costs = jnp.sum(costs, axis=0)
    min_cost = jnp.min(total_costs)
    exp_costs = jnp.exp((total_costs - min_cost) / (-temperature))
    normalizing_constant = jnp.sum(exp_costs)
    weights = exp_costs / normalizing_constant

    return jnp.tensordot(samples, weights, axes=([2], [0]))


@jax.jit(static_argnames=("planning_interval",))
@jaxtyped
def shift_control_left(
    control: Float[Array, "T D_u"],
    *,
    padding_array: Float[Array, "P D_u"],
    planning_interval: int,
) -> Float[Array, "T D_u"]:
    return jnp.concatenate([control[planning_interval:], padding_array], axis=0)
