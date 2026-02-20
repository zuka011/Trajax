from typing import Protocol, runtime_checkable
from functools import reduce

from faran.types import jaxtyped

from jaxtyping import Array as JaxArray, Float

import jax
import jax.numpy as jnp


@runtime_checkable
class HistoryWithArray(Protocol):
    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        """Returns the history as a JAX array."""
        ...


@runtime_checkable
class EstimationFilter(Protocol):
    def __call__(self, array: Float[JaxArray, "K"], /) -> Float[JaxArray, "K"]:
        """Filters out invalid estimates from the given array."""
        ...


def verify_check_recent(check_recent: int) -> None:
    assert check_recent > 0, (
        "At least one time step must be checked for invalid estimates"
    )


@jaxtyped
def invalid_obstacle_filter_from(
    *history: Float[JaxArray, "T K"] | Float[JaxArray, "T D_o K"], check_recent: int
) -> EstimationFilter:
    """Returns a filter for invalid estimates based on the given history.

    Args:
        history: The history to check for invalid estimates.
        check_recent: The number of most recent time steps to check for invalid estimates.
    """
    jax.debug.callback(verify_check_recent, check_recent)

    def invalid_mask(array: JaxArray) -> Float[JaxArray, "K"]:
        recent = array[-check_recent:]

        axes = tuple(range(recent.ndim - 1))
        return jnp.any(jnp.isnan(recent), axis=axes)

    invalid = reduce(jnp.logical_or, map(invalid_mask, history))

    def filter_invalid(array: Float[JaxArray, "K"]) -> Float[JaxArray, "K"]:
        return jnp.where(invalid, jnp.nan, array)

    return filter_invalid
