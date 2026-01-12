from typing import Protocol, Self, Any, overload

from trajax.types.mppi.common import (
    State,
    StateSequence,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
    DynamicalModel,
    Costs,
    CostFunction,
    Sampler,
    UpdateFunction,
    PaddingFunction,
    FilterFunction,
)

from jaxtyping import Array as JaxArray, Float


class JaxState[D_x: int](State[D_x], Protocol):
    @property
    def array(self) -> Float[JaxArray, "D_x"]:
        """Returns the underlying JAX array representing the state."""
        ...


class JaxStateSequence[T: int, D_x: int, StateBatchT = Any](
    StateSequence[T, D_x, StateBatchT], Protocol
):
    @property
    def array(self) -> Float[JaxArray, "T D_x"]:
        """Returns the underlying JAX array representing the state sequence."""
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
    def similar(self, *, array: Float[JaxArray, "T M"]) -> Self:
        """Creates new costs similar to this one but with the given array as its data."""
        ...

    @property
    def array(self) -> Float[JaxArray, "T M"]:
        """Returns the underlying JAX array representing the costs."""
        ...


class JaxDynamicalModel[
    StateT,
    StateSequenceT,
    StateBatchT,
    InputSequenceT,
    InputBatchT,
](
    DynamicalModel[StateT, StateSequenceT, StateBatchT, InputSequenceT, InputBatchT],
    Protocol,
): ...


class JaxSampler[InputSequenceT, InputBatchT](
    Sampler[InputSequenceT, InputBatchT], Protocol
): ...


class JaxCostFunction[InputBatchT, StateBatchT, CostsT](
    CostFunction[InputBatchT, StateBatchT, CostsT], Protocol
): ...


class JaxUpdateFunction[InputSequenceT](UpdateFunction[InputSequenceT], Protocol): ...


class JaxPaddingFunction[NominalT, PaddingT](
    PaddingFunction[NominalT, PaddingT], Protocol
): ...


class JaxFilterFunction[InputSequenceT](FilterFunction[InputSequenceT], Protocol): ...
