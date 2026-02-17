from typing import Protocol, overload, cast

from numtypes import Array, Dims

import numpy as np


class HistoryWithArray[T: int, D_o: int, K: int](Protocol):
    @property
    def array(self) -> Array[Dims[T, D_o, K]]:
        """Returns the history as a Numpy array."""
        ...


class EstimationFilter[D_o: int, K: int](Protocol):
    @overload
    def __call__(self, array: Array[Dims[K]], /) -> Array[Dims[K]]:
        """Filters out invalid estimates from the given array."""
        ...

    @overload
    def __call__(self, array: Array[Dims[D_o, K]], /) -> Array[Dims[D_o, K]]:
        """Filters out invalid estimates from the given array."""
        ...


def invalid_obstacle_filter_from[D_o: int, K: int, T: int = int](
    history: HistoryWithArray[T, D_o, K], *, check_recent: int
) -> EstimationFilter[D_o, K]:
    """Returns a filter for invalid estimates based on the given history.

    Args:
        history: The history to check for invalid estimates.
        check_recent: The number of most recent time steps to check for invalid estimates.
    """
    assert check_recent > 0, (
        "At least one time step must be checked for invalid estimates"
    )

    invalid = np.any(np.isnan(history.array[-check_recent:]), axis=(0, 1))

    def filter_invalid(array: Array) -> Array:
        return np.where(invalid, np.nan, array)

    return cast(EstimationFilter[D_o, K], filter_invalid)
