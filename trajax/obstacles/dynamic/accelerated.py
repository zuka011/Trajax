from typing import Self
from dataclasses import dataclass

from trajax.types import jaxtyped, JaxObstacleSimulator
from trajax.obstacles.accelerated import JaxObstacleStatesForTimeStep

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


@dataclass(kw_only=True)
class JaxDynamicObstacleSimulator[K: int](
    JaxObstacleSimulator[JaxObstacleStatesForTimeStep[K]]
):
    last: JaxObstacleStatesForTimeStep[K]
    velocities: Float[JaxArray, "K 2"]

    time_step: Scalar | None = None

    @staticmethod
    def create[K_: int = int](
        *,
        positions: Float[JaxArray, "K 2"],
        velocities: Float[JaxArray, "K 2"],
        obstacle_count: K_ | None = None,
    ) -> "JaxDynamicObstacleSimulator[K_]":
        headings = headings_from(velocities)
        count = positions.shape[0]

        assert obstacle_count is None or obstacle_count == count, (
            f"Expected {obstacle_count} obstacles, but got {count}."
        )

        return JaxDynamicObstacleSimulator(
            last=JaxObstacleStatesForTimeStep.create(
                x=positions[:, 0], y=positions[:, 1], heading=headings
            ),
            velocities=velocities,
        )

    def with_time_step_size(self, time_step_size: float) -> Self:
        return self.__class__(
            last=self.last,
            velocities=self.velocities,
            time_step=jnp.asarray(time_step_size),
        )

    def step(self) -> JaxObstacleStatesForTimeStep[K]:
        assert self.time_step is not None, (
            "Time step must be set to advance obstacle states."
        )

        x, y = step_obstacles(
            x=self.last.x_array,
            y=self.last.y_array,
            velocities=self.velocities,
            time_step=self.time_step,
        )

        self.last = JaxObstacleStatesForTimeStep.create(
            x=x, y=y, heading=self.last.heading_array
        )

        return self.last

    @property
    def obstacle_count(self) -> int:
        return self.last.count


@jax.jit
@jaxtyped
def step_obstacles(
    *,
    x: Float[JaxArray, "K"],
    y: Float[JaxArray, "K"],
    velocities: Float[JaxArray, "K 2"],
    time_step: Scalar,
) -> tuple[Float[JaxArray, "K"], Float[JaxArray, "K"]]:
    new_x = x + velocities[:, 0] * time_step
    new_y = y + velocities[:, 1] * time_step
    return new_x, new_y


@jax.jit
@jaxtyped
def headings_from(velocities: Float[JaxArray, "K 2"]) -> Float[JaxArray, "K"]:
    speed = jnp.linalg.norm(velocities, axis=1)
    return jnp.where(speed > 1e-6, jnp.arctan2(velocities[:, 1], velocities[:, 0]), 0.0)
