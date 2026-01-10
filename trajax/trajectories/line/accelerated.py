from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    jaxtyped,
    D_R,
    Trajectory,
    JaxPathParameters,
    JaxReferencePoints,
    JaxPositions,
    JaxLateralPositions,
    JaxLongitudinalPositions,
)

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


type Vector = Float[JaxArray, "2"]


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxLineTrajectory(
    Trajectory[
        JaxPathParameters,
        JaxReferencePoints,
        JaxPositions,
        JaxLateralPositions,
        JaxLongitudinalPositions,
    ]
):
    start: Vector
    direction: Vector

    heading: Scalar

    _path_length: Scalar

    @staticmethod
    def create(
        *, start: tuple[float, float], end: tuple[float, float], path_length: float
    ) -> "JaxLineTrajectory":
        """Generates a straight line trajectory from start to end."""
        return JaxLineTrajectory(
            start=(start_array := jnp.array(start)),
            direction=(direction := jnp.array(end) - start_array),
            heading=jnp.arctan2(direction[1], direction[0]),
            _path_length=jnp.array(path_length),
        )

    def query[T: int, M: int](
        self, parameters: JaxPathParameters[T, M]
    ) -> JaxReferencePoints[T, M]:
        return JaxReferencePoints(
            query(
                parameters=parameters.array,
                start=self.start,
                direction=self.direction,
                path_length=self._path_length,
                heading=self.heading,
            )
        )

    def lateral[T: int, M: int](
        self, positions: JaxPositions[T, M]
    ) -> JaxLateralPositions[T, M]:
        return JaxLateralPositions(
            lateral(
                positions=positions.array,
                start=self.start,
                perpendicular=self.perpendicular,
            )
        )

    def longitudinal[T: int, M: int](
        self, positions: JaxPositions[T, M]
    ) -> JaxLongitudinalPositions[T, M]:
        return JaxLongitudinalPositions(
            longitudinal(
                positions=positions.array,
                start=self.start,
                tangent=self.tangent,
                path_length=self._path_length,
                line_length=self.line_length,
            )
        )

    @property
    def path_length(self) -> float:
        return float(self._path_length)

    @cached_property
    def perpendicular(self) -> Vector:
        tangent = self.tangent
        perpendicular = jnp.array([tangent[1], -tangent[0]])
        return perpendicular

    @cached_property
    def tangent(self) -> Vector:
        return self.direction / self.line_length

    @cached_property
    def line_length(self) -> Scalar:
        return jnp.linalg.norm(self.direction)


@jax.jit
@jaxtyped
def query(
    parameters: Float[JaxArray, "T M"],
    *,
    start: Vector,
    direction: Vector,
    path_length: Scalar,
    heading: Scalar,
) -> Float[JaxArray, f"T {D_R} M"]:
    normalized = parameters / path_length
    x, y = (
        start[:, jnp.newaxis, jnp.newaxis]
        + direction[:, jnp.newaxis, jnp.newaxis] * normalized
    )
    heading = jnp.full_like(x, heading)

    return jnp.stack([x, y, heading], axis=1)


@jax.jit
@jaxtyped
def lateral(
    positions: Float[JaxArray, "T 2 M"],
    *,
    start: Vector,
    perpendicular: Vector,
) -> Float[JaxArray, "T M"]:
    relative = positions - start[:, jnp.newaxis]
    return jnp.einsum("tpm,p->tm", relative, perpendicular)


@jax.jit
@jaxtyped
def longitudinal(
    positions: Float[JaxArray, "T 2 M"],
    *,
    start: Vector,
    tangent: Vector,
    path_length: Scalar,
    line_length: Scalar,
) -> Float[JaxArray, "T M"]:
    relative = positions - start[:, jnp.newaxis]
    projection = jnp.einsum("tpm,p->tm", relative, tangent)
    return projection * path_length / line_length
