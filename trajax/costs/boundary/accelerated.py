from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    CostFunction,
    ControlInputBatch,
    JaxBoundaryDistance,
    JaxBoundaryDistanceExtractor,
    JaxCosts,
)
from trajax.states import JaxSimpleCosts

from jaxtyping import Float, Array as JaxArray, Scalar

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class JaxBoundaryCost[StateT](CostFunction[ControlInputBatch, StateT, JaxCosts]):
    distance: JaxBoundaryDistanceExtractor[StateT, JaxBoundaryDistance]
    distance_threshold: Scalar
    weight: Scalar

    @staticmethod
    def create[S](
        *,
        distance: JaxBoundaryDistanceExtractor[S, JaxBoundaryDistance],
        distance_threshold: float,
        weight: float,
    ) -> "JaxBoundaryCost":
        return JaxBoundaryCost(
            distance=distance,
            distance_threshold=jnp.array(distance_threshold),
            weight=jnp.array(weight),
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> JaxCosts[T, M]:
        return JaxSimpleCosts(
            boundary_cost(
                distance=self.distance(states=states).array,
                distance_threshold=self.distance_threshold,
                weight=self.weight,
            )
        )


@jax.jit
@jaxtyped
def boundary_cost(
    *, distance: Float[JaxArray, "T M"], distance_threshold: Scalar, weight: Scalar
) -> Float[JaxArray, "T M"]:
    cost = distance_threshold - distance
    return weight * jnp.clip(cost, 0, None)
