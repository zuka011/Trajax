from typing import Protocol, Self, overload
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
    Mppi,
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

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


class State[D_x: int = int](AnyState[D_x], Protocol):
    @property
    def array(self) -> Float[JaxArray, "D_x"]:
        """Returns the underlying JAX array representing the state."""
        ...


class StateBatch[T: int = int, D_x: int = int, M: int = int](
    AnyStateBatch[T, D_x, M], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_x M"]:
        """Returns the underlying JAX array representing the state batch."""
        ...


class ControlInputSequence[T: int = int, D_u: int = int](
    AnyControlInputSequence[T, D_u], Protocol
):
    @overload
    def similar(self, *, array: Float[JaxArray, "T D_u"]) -> Self:
        """Creates a new control input sequence similar to this one but with the given
        array as its data.
        """
        ...

    @overload
    def similar[L: int](
        self, *, array: Float[JaxArray, "L D_u"], length: L
    ) -> "ControlInputSequence[L, D_u]":
        """Returns a new control input sequence similar to this one but using the provided array.
        The length of the new sequence may differ from the original."""
        ...

    @property
    def array(self) -> Float[JaxArray, "T D_u"]:
        """Returns the underlying JAX array representing the control input sequence."""
        ...


class ControlInputBatch[T: int = int, D_u: int = int, M: int = int](
    AnyControlInputBatch[T, D_u, M], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_u M"]:
        """Returns the underlying JAX array representing the control input batch."""
        ...


class Costs[T: int = int, M: int = int](AnyCosts[T, M], Protocol):
    @property
    def array(self) -> Float[JaxArray, "T M"]:
        """Returns the underlying JAX array representing the costs."""
        ...


type DynamicalModel[
    InStateT: State,
    OutStateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
] = AnyDynamicalModel[
    InStateT, OutStateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT
]

type Sampler[SequenceT: ControlInputSequence, BatchT: ControlInputBatch] = AnySampler[
    SequenceT, BatchT
]

type CostFunction[
    ControlInputBatchT: ControlInputBatch,
    StateBatchT: StateBatch,
    CostsT: Costs,
] = AnyCostFunction[ControlInputBatchT, StateBatchT, CostsT]

type UpdateFunction[C: ControlInputSequence] = AnyUpdateFunction[C]

type PaddingFunction[N: ControlInputSequence, P: ControlInputSequence] = (
    AnyPaddingFunction[N, P]
)

type FilterFunction[C: ControlInputSequence] = AnyFilterFunction[C]


class ZeroPadding[T: int = int, D_u: int = int, L: int = int]:
    def __call__(
        self, *, nominal_input: ControlInputSequence[T, D_u], padding_size: L
    ) -> ControlInputSequence[L, D_u]:
        padding_array = jnp.zeros((padding_size, nominal_input.array.shape[1]))
        return nominal_input.similar(array=padding_array, length=padding_size)


@dataclass(kw_only=True, frozen=True)
class JaxMppi[
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
    ) -> "JaxMppi[IS, OS, SB, CIS, CIP, CIB, C]":
        """Creates a JAX-based MPPI controller."""
        return JaxMppi(
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
        cost_function: CostFunction[ControlInputBatchT, StateBatchT, CostsT],
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
