from typing import Sequence

from faran.types import NumPyCosts, CostSumFunction

from numtypes import shape_of

import numpy as np


class NumPyCostSumFunction[CostsT: NumPyCosts](CostSumFunction[CostsT]):
    """Sums multiple cost arrays element-wise into a single cost."""

    def __call__(self, costs: Sequence[CostsT], *, initial: CostsT) -> CostsT:
        total = np.sum([it.array for it in costs], axis=0) + initial.array

        assert shape_of(
            total, matches=(initial.horizon, initial.rollout_count), name="summed costs"
        )

        return initial.similar(array=total)
