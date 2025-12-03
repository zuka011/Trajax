from typing import Protocol

from trajax.type import jaxtyped
from trajax.mppi.common import (
    Control,
    ControlInputSequence as AnyControlInputSequence,
    ControlInputBatch as AnyControlInputBatch,
    State as AnyState,
    StateBatch as AnyStateBatch,
    DynamicalModel as AnyDynamicalModel,
    Sampler as AnySampler,
    CostFunction as AnyCostFunction,
)

from jaxtyping import Array, Float, Scalar

import jax
import jax.numpy as jnp


class JaxMppi:
    class ControlInputSequence[T: int, D_u: int](
        AnyControlInputSequence[T, D_u], Protocol
    ):
        def similar(
            self, *, array: Float[Array, "T D_u"]
        ) -> "JaxMppi.ControlInputSequence":
            """Returns a new control input sequence similar to this one but using the provided array."""
            ...

        @property
        def array(self) -> Float[Array, "T D_u"]:
            """Returns the underlying JAX array representing the control input sequence."""
            ...

    class ControlInputBatch[T: int, D_u: int, M: int](
        AnyControlInputBatch[T, D_u, M], Protocol
    ):
        @property
        def array(self) -> Float[Array, "T D_u M"]:
            """Returns the underlying JAX array representing the control input batch."""
            ...

    class State[D_x: int](AnyState[D_x], Protocol):
        @property
        def array(self) -> Float[Array, "D_x"]:
            """Returns the underlying JAX array representing the state."""
            ...

    class StateBatch[T: int, D_x: int, M: int](AnyStateBatch[T, D_x, M], Protocol):
        @property
        def array(self) -> Float[Array, "T D_x M"]:
            """Returns the underlying JAX array representing the state batch."""
            ...

    type DynamicalModel[T: int, D_u: int, D_x: int, M: int] = AnyDynamicalModel[
        JaxMppi.ControlInputBatch[T, D_u, M],
        JaxMppi.State[D_x],
        JaxMppi.StateBatch[T, D_x, M],
    ]

    type Sampler[T: int, D_u: int, M: int] = AnySampler[
        JaxMppi.ControlInputSequence[T, D_u], JaxMppi.ControlInputBatch[T, D_u, M]
    ]

    type CostFunction[T: int, D_u: int, D_x: int, M: int] = AnyCostFunction[
        JaxMppi.ControlInputBatch[T, D_u, M],
        JaxMppi.StateBatch[T, D_x, M],
        Float[Array, "T M"],
    ]

    @staticmethod
    def create() -> "JaxMppi":
        return JaxMppi()

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

        return Control(optimal=nominal_input.similar(array=optimal_control))


@jax.jit
@jaxtyped
def compute_weighted_control(
    samples: Float[Array, "T D_u M"], costs: Float[Array, "T M"], temperature: Scalar
) -> Float[Array, "T D_u"]:
    total_costs = jnp.sum(costs, axis=0)
    min_cost = jnp.min(total_costs)
    weights = jnp.exp((total_costs - min_cost) / (-temperature))

    return jnp.tensordot(samples, weights, axes=([2], [0]))
