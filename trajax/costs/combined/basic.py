from typing import Sequence

from trajax.types import NumPyCosts, CostSumFunction

from numtypes import shape_of

import numpy as np


class NumPyCostSumFunction[CostsT: NumPyCosts](CostSumFunction[CostsT]):
    def __call__(self, costs: Sequence[CostsT], *, initial: CostsT) -> CostsT:
        total = np.sum([it.array for it in costs], axis=0) + initial.array

        assert shape_of(
            total, matches=(initial.horizon, initial.rollout_count), name="summed costs"
        )

        return initial.similar(array=total)
