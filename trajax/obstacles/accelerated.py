from dataclasses import dataclass

from trajax.costs.distance import JaxObstacleStateProvider, JaxObstaclePositions

from numtypes import D
from jaxtyping import Array as JaxArray, Float

import jax.numpy as jnp


@dataclass(frozen=True)
class JaxStaticObstacleStateProvider[T: int, K: int](
    JaxObstacleStateProvider[T, D[2], K]
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
        positions: Float[JaxArray, "K 2"],
        *,
        horizon: T_,
        obstacle_count: K_ | None = None,
    ) -> "JaxStaticObstacleStateProvider[T_, K_]":
        K = obstacle_count if obstacle_count is not None else positions.shape[0]
        x = jnp.tile(positions[:, 0], (horizon, 1))
        y = jnp.tile(positions[:, 1], (horizon, 1))

        assert x.shape == y.shape == (horizon, K), (
            f"Expected shapes {(horizon, K)}, but got x with shape {x.shape} and y with shape {y.shape}."
        )

        return JaxStaticObstacleStateProvider(JaxObstaclePositions.create(x=x, y=y))

    def __call__(self) -> JaxObstaclePositions[T, K]:
        return self.positions
