from dataclasses import dataclass

from faran.types import (
    jaxtyped,
    CostFunction,
    ControlInputBatch,
    Trajectory,
    BoundaryPoints,
    BoundaryWidthsDescription,
    JaxReferencePoints,
    JaxBoundaryDistance,
    JaxBoundaryDistanceExtractor,
    JaxCosts,
    JaxPathParameters,
    JaxPositions,
    JaxLateralPositions,
    JaxPositionExtractor,
)
from faran.states import JaxSimpleCosts

from jaxtyping import Float, Array as JaxArray, Scalar

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class JaxBoundaryCost[StateT](CostFunction[ControlInputBatch, StateT, JaxCosts]):
    """Penalizes states that approach within a threshold distance of the corridor edges."""

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
    """Fixed-width corridor boundary around a reference trajectory."""

    reference: Trajectory[
        JaxPathParameters, JaxReferencePoints, JaxPositions, JaxLateralPositions
    ]
    position_extractor: JaxPositionExtractor[StateT]
    _left: Scalar
    _right: Scalar

    @staticmethod
    def create[S](
        *,
        reference: Trajectory[
            JaxPathParameters, JaxReferencePoints, JaxPositions, JaxLateralPositions
        ],
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
            _left=jnp.array(left),
            _right=jnp.array(right),
        )

    def __call__(self, *, states: StateT) -> JaxBoundaryDistance:
        positions = self.position_extractor(states)
        lateral = self.reference.lateral(positions)

        return JaxBoundaryDistance(
            boundary_distance(lateral=lateral.array, left=self._left, right=self._right)
        )

    def left[L: int](self, *, sample_count: L = 100) -> BoundaryPoints[L]:
        s = JaxPathParameters(
            jnp.linspace(0, self.reference.path_length, sample_count).reshape(-1, 1)
        )

        return np.asarray(
            (
                self.reference.query(s).positions_array
                - self._left * self.reference.normal(s).array
            )[..., 0]
        )

    def right[L: int](self, *, sample_count: L = 100) -> BoundaryPoints[L]:
        s = JaxPathParameters(
            jnp.linspace(0, self.reference.path_length, sample_count).reshape(-1, 1)
        )

        return np.asarray(
            (
                self.reference.query(s).positions_array
                + self._right * self.reference.normal(s).array
            )[..., 0]
        )


@dataclass(kw_only=True, frozen=True)
class JaxPiecewiseFixedWidthBoundary[StateT](
    JaxBoundaryDistanceExtractor[StateT, JaxBoundaryDistance]
):
    """Piecewise fixed-width corridor boundary with segment-varying widths."""

    reference: Trajectory[
        JaxPathParameters, JaxReferencePoints, JaxPositions, JaxLateralPositions
    ]
    position_extractor: JaxPositionExtractor[StateT]
    breakpoints: Float[JaxArray, "B"]
    left_widths: Float[JaxArray, "B"]
    right_widths: Float[JaxArray, "B"]

    @staticmethod
    def create[S](
        *,
        reference: Trajectory[
            JaxPathParameters, JaxReferencePoints, JaxPositions, JaxLateralPositions
        ],
        position_extractor: JaxPositionExtractor[S],
        widths: BoundaryWidthsDescription,
    ) -> "JaxPiecewiseFixedWidthBoundary[S]":
        """Creates a piecewise fixed-width boundary distance extractor.

        This component assumes a piecewise constant corridor width around a reference trajectory.
        Different segments of the trajectory can have different left and right widths.

        Args:
            reference: The reference trajectory defining the center of the corridor.
            position_extractor: Function to extract positions from states.
            widths: A mapping from longitudinal breakpoints to {"left": float, "right": float}
                   dictionaries defining the corridor widths for each segment.
        """
        sorted_breakpoints = sorted(widths.keys())

        return JaxPiecewiseFixedWidthBoundary(
            reference=reference,
            position_extractor=position_extractor,
            breakpoints=jnp.array(sorted_breakpoints),
            left_widths=jnp.array([widths[s]["left"] for s in sorted_breakpoints]),
            right_widths=jnp.array([widths[s]["right"] for s in sorted_breakpoints]),
        )

    def __call__(self, *, states: StateT) -> JaxBoundaryDistance:
        positions = self.position_extractor(states)
        lateral = self.reference.lateral(positions)
        longitudinal = self.reference.longitudinal(positions)

        return JaxBoundaryDistance(
            piecewise_boundary_distance(
                lateral=lateral.array,
                longitudinal=longitudinal.array,
                breakpoints=self.breakpoints,
                left_widths=self.left_widths,
                right_widths=self.right_widths,
            )
        )

    def left[L: int](self, *, sample_count: L = 100) -> BoundaryPoints[L]:
        s_values = jnp.linspace(0, self.reference.path_length, sample_count)
        s = JaxPathParameters(s_values.reshape(-1, 1))

        return np.asarray(
            piecewise_left_boundary_points(
                positions=self.reference.query(s).positions_array,
                normals=self.reference.normal(s).array,
                s_values=s_values,
                breakpoints=self.breakpoints,
                left_widths=self.left_widths,
            )
        )

    def right[L: int](self, *, sample_count: L = 100) -> BoundaryPoints[L]:
        s_values = jnp.linspace(0, self.reference.path_length, sample_count)
        s = JaxPathParameters(s_values.reshape(-1, 1))

        return np.asarray(
            piecewise_right_boundary_points(
                positions=self.reference.query(s).positions_array,
                normals=self.reference.normal(s).array,
                s_values=s_values,
                breakpoints=self.breakpoints,
                right_widths=self.right_widths,
            )
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


@jax.jit
@jaxtyped
def piecewise_boundary_distance(
    *,
    lateral: Float[JaxArray, "T M"],
    longitudinal: Float[JaxArray, "T M"],
    breakpoints: Float[JaxArray, "N"],
    left_widths: Float[JaxArray, "N"],
    right_widths: Float[JaxArray, "N"],
) -> Float[JaxArray, "T M"]:
    segment_indices = jnp.searchsorted(breakpoints, longitudinal, side="right") - 1
    segment_indices = jnp.clip(segment_indices, 0, len(breakpoints) - 1)

    left = left_widths[segment_indices]
    right = right_widths[segment_indices]

    distance_to_left = left + lateral
    distance_to_right = right - lateral

    return jnp.minimum(distance_to_left, distance_to_right)


@jax.jit
@jaxtyped
def piecewise_left_boundary_points(
    *,
    positions: Float[JaxArray, "N 2 1"],
    normals: Float[JaxArray, "N 2 1"],
    s_values: Float[JaxArray, "N"],
    breakpoints: Float[JaxArray, "K"],
    left_widths: Float[JaxArray, "K"],
) -> Float[JaxArray, "N 2"]:
    segment_indices = jnp.searchsorted(breakpoints, s_values, side="right") - 1
    segment_indices = jnp.clip(segment_indices, 0, len(breakpoints) - 1)
    left = left_widths[segment_indices]

    return (positions - left[:, jnp.newaxis, jnp.newaxis] * normals)[..., 0]


@jax.jit
@jaxtyped
def piecewise_right_boundary_points(
    *,
    positions: Float[JaxArray, "N 2 1"],
    normals: Float[JaxArray, "N 2 1"],
    s_values: Float[JaxArray, "N"],
    breakpoints: Float[JaxArray, "K"],
    right_widths: Float[JaxArray, "K"],
) -> Float[JaxArray, "N 2"]:
    segment_indices = jnp.searchsorted(breakpoints, s_values, side="right") - 1
    segment_indices = jnp.clip(segment_indices, 0, len(breakpoints) - 1)
    right = right_widths[segment_indices]

    return (positions + right[:, jnp.newaxis, jnp.newaxis] * normals)[..., 0]
