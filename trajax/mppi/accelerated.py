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
    NoUpdate,
    UpdateFunction as AnyUpdateFunction,
    Sampler as AnySampler,
    CostFunction as AnyCostFunction,
)

from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp


class JaxZeroPadding:
    pass


class ControlInputSequence[T: int, D_u: int](AnyControlInputSequence[T, D_u], Protocol):
    def similar(
        self, *, array: Float[Array, "T D_u"]
    ) -> "ControlInputSequence[T, D_u]":
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

    type CostFunction[T: int, D_u: int, D_x: int, M: int] = AnyCostFunction[
        "JaxMppi.ControlInputBatch[T, D_u, M]",
        "JaxMppi.StateBatch[T, D_x, M]",
        Float[Array, "T M"],
    ]

    planning_interval: int
    update_function: UpdateFunction

    @staticmethod
    def create(
        *,
        planning_interval: int = 1,
        update_function: UpdateFunction = NoUpdate(),
        padding_function: ... = None,
    ) -> "JaxMppi":
        """Creates a JAX-based MPPI controller."""
        return JaxMppi(
            planning_interval=planning_interval, update_function=update_function
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

        optimal_control = compute_weighted_control(samples.array, costs, temperature)
        control_dimension = optimal_control.shape[1]

        optimal_input = nominal_input.similar(array=optimal_control)
        nominal_input = self.update_function(
            nominal_input=nominal_input, optimal_input=optimal_input
        )

        shifted_control = shift_control_left(
            nominal_input.array,
            planning_interval=self.planning_interval,
            control_dimension=control_dimension,
        )

        return Control(
            optimal=optimal_input, nominal=nominal_input.similar(array=shifted_control)
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


@jax.jit(static_argnames=("planning_interval", "control_dimension"))
@jaxtyped
def shift_control_left(
    control: Float[Array, "T D_u"], *, planning_interval: int, control_dimension: int
) -> Float[Array, "T D_u"]:
    zeros = jnp.zeros((planning_interval, control_dimension), dtype=control.dtype)
    return jnp.concatenate([control[planning_interval:], zeros], axis=0)
