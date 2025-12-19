from typing import Sequence

from trajax.mppi import (
    NumPyCosts,
)
from trajax.costs.combined.common import CostSumFunction

from numtypes import shape_of

import numpy as np


class NumPyCostSumFunction[CostsT: NumPyCosts](CostSumFunction[CostsT]):
    def __call__(self, costs: Sequence[CostsT], *, initial: CostsT) -> CostsT:
        total = np.sum(costs, axis=0) + np.asarray(initial)

        assert shape_of(
            total, matches=(initial.horizon, initial.rollout_count), name="summed costs"
        )

        return initial.similar(array=total)
