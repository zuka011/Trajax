from typing import Sequence

from trajax.types import jaxtyped, JaxCosts, CostSumFunction

from jaxtyping import Array as JaxArray, Float

import jax
import jax.numpy as jnp


class JaxCostSumFunction[CostsT: JaxCosts](CostSumFunction[CostsT]):
    def __call__(self, costs: Sequence[CostsT], *, initial: CostsT) -> CostsT:
        return initial.similar(
            array=sum_costs(costs=[it.array for it in costs], initial=initial.array)
        )


@jax.jit
@jaxtyped
def sum_costs(
    costs: list[Float[JaxArray, "T M"]], initial: Float[JaxArray, "T M"]
) -> Float[JaxArray, "T M"]:
    return jnp.stack(costs).sum(axis=0) + initial
