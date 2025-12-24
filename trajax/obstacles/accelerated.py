from dataclasses import dataclass

from trajax.costs import JaxObstacleStateProvider
from trajax.costs.distance import JaxObstaclePositions

from numtypes import D
from jaxtyping import Array as JaxArray, Float

import jax.numpy as jnp


@dataclass(frozen=True)
class JaxStaticObstacleStateProvider[T: int, K: int](
    JaxObstacleStateProvider[JaxObstaclePositions[T, K]]
):
    positions: JaxObstaclePositions[T, K]

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
            JaxObstaclePositions.create(x=x, y=y, heading=heading)
        )

    def __call__(self) -> JaxObstaclePositions[T, K]:
        return self.positions
