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

from numtypes import Array, Dims


class NumPyState[D_x: int](State[D_x], Protocol):
    @property
    def array(self) -> Array[Dims[D_x]]:
        """Returns the underlying NumPy array representing the state."""
        ...


class NumPyStateSequence[T: int, D_x: int, StateBatchT = Any](
    StateSequence[T, D_x, StateBatchT], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_x]]:
        """Returns the underlying NumPy array representing the state sequence."""
        ...


class NumPyStateBatch[T: int, D_x: int, M: int](StateBatch[T, D_x, M], Protocol):
    @property
    def array(self) -> Array[Dims[T, D_x, M]]:
        """Returns the underlying NumPy array representing the state batch."""
        ...


class NumPyControlInputSequence[T: int, D_u: int](
    ControlInputSequence[T, D_u], Protocol
):
    @overload
    def similar(self, *, array: Array[Dims[T, D_u]]) -> Self:
        """Creates a new control input sequence similar to this one but with the given
        array as its data.
        """
        ...

    @overload
    def similar[L: int](
        self, *, array: Array[Dims[L, D_u]], length: L
    ) -> "NumPyControlInputSequence[L, D_u]":
        """Creates a new control input sequence similar to this one but with the given
        array as its data. The length of the new sequence may differ from the original.
        """
        ...

    @property
    def array(self) -> Array[Dims[T, D_u]]:
        """Returns the underlying NumPy array representing the control input sequence."""
        ...


class NumPyControlInputBatch[T: int, D_u: int, M: int](
    ControlInputBatch[T, D_u, M], Protocol
):
    @property
    def array(self) -> Array[Dims[T, D_u, M]]:
        """Returns the underlying NumPy array representing the control input batch."""
        ...


class NumPyCosts[T: int, M: int](Costs[T, M], Protocol):
    def similar(self, *, array: Array[Dims[T, M]]) -> Self:
        """Creates new costs similar to this one but with the given array as its data."""
        ...

    @property
    def array(self) -> Array[Dims[T, M]]:
        """Returns the underlying NumPy array representing the costs."""
        ...


class NumPyDynamicalModel[
    StateT,
    StateSequenceT,
    StateBatchT,
    InputSequenceT,
    InputBatchT,
](
    DynamicalModel[StateT, StateSequenceT, StateBatchT, InputSequenceT, InputBatchT],
    Protocol,
): ...


class NumPySampler[InputSequenceT, InputBatchT](
    Sampler[InputSequenceT, InputBatchT], Protocol
): ...


class NumPyCostFunction[InputBatchT, StateBatchT, CostsT](
    CostFunction[InputBatchT, StateBatchT, CostsT], Protocol
): ...


class NumPyUpdateFunction[InputSequenceT](UpdateFunction[InputSequenceT], Protocol): ...


class NumPyPaddingFunction[NominalT, PaddingT](
    PaddingFunction[NominalT, PaddingT], Protocol
): ...


class NumPyFilterFunction[InputSequenceT](FilterFunction[InputSequenceT], Protocol): ...
