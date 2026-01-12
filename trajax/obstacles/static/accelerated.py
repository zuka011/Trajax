from typing import Self
from dataclasses import dataclass

from trajax.types import JaxObstacleSimulator
from trajax.obstacles.accelerated import JaxObstacleStatesForTimeStep

from numtypes import Array, Dims, D
from jaxtyping import Array as JaxArray, Float

import jax.numpy as jnp


@dataclass(frozen=True)
class JaxStaticObstacleSimulator[K: int](
    JaxObstacleSimulator[JaxObstacleStatesForTimeStep[K]]
):
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

    def step(self) -> JaxObstacleStatesForTimeStep[K]:
        return JaxObstacleStatesForTimeStep.create(
            x=self.positions[:, 0], y=self.positions[:, 1], heading=self.headings
        )

    @property
    def obstacle_count(self) -> int:
        return self.positions.shape[0]
