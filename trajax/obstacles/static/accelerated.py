from typing import Self
from dataclasses import dataclass

from trajax.types import JaxObstacleStateProvider
from trajax.predictors import StaticPredictor
from trajax.obstacles.history import JaxObstacleIds, JaxObstacleStatesRunningHistory
from trajax.obstacles.accelerated import JaxObstacleStates, JaxObstacleStatesForTimeStep
from trajax.obstacles.common import PredictingObstacleStateProvider

from numtypes import D
from jaxtyping import Array as JaxArray, Float

import jax.numpy as jnp


@dataclass(frozen=True)
class JaxStaticObstacleStateProvider[T: int, K: int](
    JaxObstacleStateProvider[JaxObstacleStates[T, K]]
):
    inner: PredictingObstacleStateProvider[
        JaxObstacleStatesForTimeStep[K],
        JaxObstacleIds[K],
        JaxObstacleStates[int, K],
        JaxObstacleStates[T, K],
    ]

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
        count = positions.shape[0]
        headings = headings if headings is not None else jnp.zeros(shape=(count,))

        return JaxStaticObstacleStateProvider(  # type: ignore
            inner=PredictingObstacleStateProvider.create(
                predictor=StaticPredictor.create(horizon=horizon),
                history=JaxObstacleStatesRunningHistory.single(
                    JaxObstacleStatesForTimeStep.create(
                        x=positions[:, 0], y=positions[:, 1], heading=headings
                    ),
                ),
            ),
        )

    def with_time_step(self, time_step: float) -> Self:
        # Time step does not matter.
        return self

    def with_predictor(self, predictor: object) -> Self:
        # Predictor does not matter.
        return self

    def __call__(self) -> JaxObstacleStates[T, K]:
        return self.inner()

    def step(self) -> None:
        # Nothing to do, since the obstacles are static.
        pass
