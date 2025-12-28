from typing import Protocol

from jaxtyping import Array as JaxArray, Float


class JaxControlInputBatchCreator[InputBatchT](Protocol):
    def __call__(self, *, array: Float[JaxArray, "T D_u M"]) -> InputBatchT:
        """Creates a control input batch from the given array."""
        ...
