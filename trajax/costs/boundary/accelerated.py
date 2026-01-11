from typing import Any
from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    CostFunction,
    ControlInputBatch,
    Trajectory,
    JaxBoundaryDistance,
    JaxBoundaryDistanceExtractor,
    JaxCosts,
    JaxPositions,
    JaxLateralPositions,
    JaxPositionExtractor,
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
    ) -> "JaxBoundaryCost[S]":
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


@dataclass(kw_only=True, frozen=True)
class JaxFixedWidthBoundary[StateT](
    JaxBoundaryDistanceExtractor[StateT, JaxBoundaryDistance]
):
    reference: Trajectory[Any, Any, JaxPositions, JaxLateralPositions]
    position_extractor: JaxPositionExtractor[StateT]
    left: Scalar
    right: Scalar

    @staticmethod
    def create[S](
        *,
        reference: Trajectory[Any, Any, JaxPositions, JaxLateralPositions],
        position_extractor: JaxPositionExtractor[S],
        left: float,
        right: float,
    ) -> "JaxFixedWidthBoundary[S]":
        """Creates a fixed-width boundary distance extractor.

        This component assumes a fixed-width corridor around a reference trajectory. The left and
        right widths can be different (asymmetric corridor).

        Args:
            reference: The reference trajectory defining the center of the corridor.
            position_extractor: Function to extract positions from states.
            left: The width of the left side of the corridor.
            right: The width of the right side of the corridor.
        """
        assert left >= -right, (
            f"The boundaries appear to be inverted. Left: {left}, Right: {right}. "
            f"Make sure the total width (left + right) is non-negative, got {left + right}."
        )

        return JaxFixedWidthBoundary(
            reference=reference,
            position_extractor=position_extractor,
            left=jnp.array(left),
            right=jnp.array(right),
        )

    def __call__(self, *, states: StateT) -> JaxBoundaryDistance:
        positions = self.position_extractor(states)
        lateral = self.reference.lateral(positions)

        return JaxBoundaryDistance(
            boundary_distance(lateral=lateral.array, left=self.left, right=self.right)
        )


@jax.jit
@jaxtyped
def boundary_cost(
    *, distance: Float[JaxArray, "T M"], distance_threshold: Scalar, weight: Scalar
) -> Float[JaxArray, "T M"]:
    cost = distance_threshold - distance
    return weight * jnp.clip(cost, 0, None)


@jax.jit
@jaxtyped
def boundary_distance(
    *, lateral: Float[JaxArray, "T M"], left: Scalar, right: Scalar
) -> Float[JaxArray, "T M"]:
    distance_to_left = left + lateral
    distance_to_right = right - lateral
    return jnp.minimum(distance_to_left, distance_to_right)
