from typing import Self, Any
from dataclasses import dataclass

from trajax.types import JaxObstacleStateProvider
from trajax.obstacles.accelerated import JaxObstacleStates

from numtypes import D
from jaxtyping import Array as JaxArray, Float

import jax.numpy as jnp


@dataclass(frozen=True)
class JaxStaticObstacleStateProvider[T: int, K: int](
    JaxObstacleStateProvider[JaxObstacleStates[T, K]]
):
    states: JaxObstacleStates[T, K]

    @staticmethod
    def empty[T_: int](*, horizon: T_) -> "JaxStaticObstacleStateProvider[T_, D[0]]":
        positions = jnp.empty((0, 2))

        return JaxStaticObstacleStateProvider.create(
            positions=positions, horizon=horizon
        )

    @staticmethod
    def create[T_: int, K_: int](
        *,
        positions: Float[JaxArray, "K 2"],
        headings: Float[JaxArray, "K"] | None = None,
        horizon: T_,
        obstacle_count: K_ | None = None,
    ) -> "JaxStaticObstacleStateProvider[T_, K_]":
        K = obstacle_count if obstacle_count is not None else positions.shape[0]
        x = jnp.tile(positions[:, 0], (horizon, 1))
        y = jnp.tile(positions[:, 1], (horizon, 1))

        if headings is not None:
            heading = jnp.tile(headings, (horizon, 1))
        else:
            heading = jnp.zeros((horizon, K))

        assert x.shape == y.shape == heading.shape == (horizon, K), (
            f"Expected shapes {(horizon, K)}, but got x with shape {x.shape}, y with shape {y.shape}, heading with shape {heading.shape}."
        )

        return JaxStaticObstacleStateProvider(
            JaxObstacleStates.create(x=x, y=y, heading=heading)
        )

    def with_time_step(self, time_step: float) -> Self:
        return self

    def with_predictor(self, predictor: Any) -> Self:
        return self

    def __call__(self) -> JaxObstacleStates[T, K]:
        return self.states

    def step(self) -> None:
        pass
