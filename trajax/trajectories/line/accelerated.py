from dataclasses import dataclass

from trajax.types import (
    jaxtyped,
    D_R,
    Trajectory,
    JaxPathParameters,
    JaxReferencePoints,
)

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


@dataclass(kw_only=True, frozen=True)
class JaxLineTrajectory(Trajectory[JaxPathParameters, JaxReferencePoints]):
    start: tuple[float, float]
    end: tuple[float, float]

    delta_x: float
    delta_y: float
    length: float
    heading: float

    @staticmethod
    def create(
        *, start: tuple[float, float], end: tuple[float, float], path_length: float
    ) -> "JaxLineTrajectory":
        """Generates a straight line trajectory from start to end."""
        return JaxLineTrajectory(
            start=start,
            end=end,
            delta_x=(delta_x := end[0] - start[0]),
            delta_y=(delta_y := end[1] - start[1]),
            length=path_length,
            heading=float(jnp.arctan2(delta_y, delta_x)),
        )

    def query[T: int, M: int](
        self, parameters: JaxPathParameters[T, M]
    ) -> JaxReferencePoints[T, M]:
        return JaxReferencePoints(
            query(
                parameters=parameters.array,
                length=self.length,
                delta_x=self.delta_x,
                delta_y=self.delta_y,
                start=self.start,
                heading=self.heading,
            )
        )

    @property
    def path_length(self) -> float:
        return self.length


@jax.jit
@jaxtyped
def query(
    parameters: Float[JaxArray, "T M"],
    *,
    length: Scalar,
    delta_x: Scalar,
    delta_y: Scalar,
    start: tuple[Scalar, Scalar],
    heading: Scalar,
) -> Float[JaxArray, f"T {D_R} M"]:
    normalized = parameters / length
    x = start[0] + normalized * delta_x
    y = start[1] + normalized * delta_y
    heading = jnp.full_like(x, heading)

    return stack(x=x, y=y, heading=heading)


@jax.jit
@jaxtyped
def stack(
    *,
    x: Float[JaxArray, "T M"],
    y: Float[JaxArray, "T M"],
    heading: Float[JaxArray, "T M"],
) -> Float[JaxArray, f"T {D_R} M"]:
    return jnp.stack([x, y, heading], axis=-1).transpose(0, 2, 1)
