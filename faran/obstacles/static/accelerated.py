from typing import Self
from dataclasses import dataclass

from faran.types import JaxObstacleSimulator
from faran.obstacles.accelerated import JaxObstacle2dPosesForTimeStep

from numtypes import Array, Dims, D
from jaxtyping import Array as JaxArray, Float

import jax.numpy as jnp


@dataclass(frozen=True)
class JaxStaticObstacleSimulator[K: int](
    JaxObstacleSimulator[JaxObstacle2dPosesForTimeStep[K]]
):
    """Simulates stationary obstacles by replicating fixed positions over the horizon."""

    positions: Float[JaxArray, "K 2"]
    headings: Float[JaxArray, "K"]

    @staticmethod
    def empty() -> "JaxStaticObstacleSimulator[D[0]]":
        positions = jnp.empty((0, 2))

        return JaxStaticObstacleSimulator.create(positions=positions)

    @staticmethod
    def create[K_: int = int](
        *,
        positions: Array[Dims[K_, D[2]]] | Float[JaxArray, "K 2"],
        headings: Array[Dims[K_]] | Float[JaxArray, "K"] | None = None,
        obstacle_count: K_ | None = None,
    ) -> "JaxStaticObstacleSimulator[K_]":
        count = positions.shape[0]

        assert obstacle_count is None or obstacle_count == count, (
            f"Expected {obstacle_count} obstacles, but got {count}."
        )

        headings = headings if headings is not None else jnp.zeros(shape=(count,))

        return JaxStaticObstacleSimulator(
            positions=jnp.asarray(positions), headings=jnp.asarray(headings)
        )

    def with_time_step_size(self, time_step_size: float) -> Self:
        # Time step does not matter.
        return self

    def step(self) -> JaxObstacle2dPosesForTimeStep[K]:
        return JaxObstacle2dPosesForTimeStep.create(
            x=self.positions[:, 0], y=self.positions[:, 1], heading=self.headings
        )

    @property
    def obstacle_count(self) -> int:
        return self.positions.shape[0]
