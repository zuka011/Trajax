from typing import Self, Final, Never, cast
from dataclasses import dataclass

from trajax.types import jaxtyped, JaxObstacleStateProvider, ObstacleMotionPredictor
from trajax.obstacles.assignment import JaxHungarianObstacleIdAssignment
from trajax.obstacles.history import JaxObstacleIds, JaxObstacleStatesRunningHistory
from trajax.obstacles.accelerated import (
    JaxObstacleStates,
    JaxObstacleStatesForTimeStep,
    JaxObstacle2dPositions,
    JaxObstacle2dPositionsForTimeStep,
)
from trajax.obstacles.common import PredictingObstacleStateProvider

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


HISTORY_HORIZON: Final = 3


class ObstaclePositionExtractor:
    def of_states_for_time_step[K: int](
        self, states: JaxObstacleStatesForTimeStep[K], /
    ) -> JaxObstacle2dPositionsForTimeStep[K]:
        return states.positions()

    def of_states[K: int](
        self, states: JaxObstacleStates[int, K], /
    ) -> JaxObstacle2dPositions[int, K]:
        return states.positions()


@dataclass(kw_only=True)
class JaxDynamicObstacleStateProvider[PredictionT, K: int](
    JaxObstacleStateProvider[PredictionT]
):
    type MotionPredictor[P, K_: int] = ObstacleMotionPredictor[
        JaxObstacleStates[int, K_], P
    ]

    history: JaxObstacleStatesRunningHistory[int, K]
    velocities: Float[JaxArray, "K 2"]

    time_step: Scalar | None = None
    inner: (
        PredictingObstacleStateProvider[
            JaxObstacleStatesForTimeStep[K],
            JaxObstacleIds[K],
            JaxObstacleStates[int, K],
            PredictionT,
        ]
        | None
    ) = None

    @staticmethod
    def create[K_: int = int](
        *,
        positions: Float[JaxArray, "K 2"],
        velocities: Float[JaxArray, "K 2"],
        obstacle_count: K_ | None = None,
    ) -> "JaxDynamicObstacleStateProvider[Never, K_]":
        headings = headings_from(velocities)
        obstacle_count = (
            obstacle_count
            if obstacle_count is not None
            else cast(K_, positions.shape[0])
        )

        assert positions.shape == (obstacle_count, 2), (
            f"Positions must have shape ({obstacle_count}, 2), "
            f"but got {positions.shape}."
        )
        assert velocities.shape == (obstacle_count, 2), (
            f"Velocities must have shape ({obstacle_count}, 2), "
            f"but got {velocities.shape}."
        )

        return JaxDynamicObstacleStateProvider(
            history=JaxObstacleStatesRunningHistory.single(
                JaxObstacleStatesForTimeStep.create(
                    x=positions[:, 0], y=positions[:, 1], heading=headings
                ),
                horizon=HISTORY_HORIZON,
                obstacle_count=obstacle_count,
            ),
            velocities=velocities,
        )

    def with_time_step(self, time_step: float) -> Self:
        return self.__class__(
            history=self.history,
            velocities=self.velocities,
            time_step=jnp.asarray(time_step),
            inner=self.inner,
        )

    def with_predictor[P](
        self, predictor: MotionPredictor[P, K]
    ) -> "JaxDynamicObstacleStateProvider[P, K]":
        return JaxDynamicObstacleStateProvider(
            history=self.history,
            velocities=self.velocities,
            time_step=self.time_step,
            inner=PredictingObstacleStateProvider.create(
                predictor=predictor,
                history=self.history,
                id_assignment=JaxHungarianObstacleIdAssignment.create(
                    position_extractor=ObstaclePositionExtractor(), cutoff=1.0
                ),
            ),
        )

    def __call__(self) -> PredictionT:
        assert self.inner is not None, (
            "Motion predictor must be set to provide obstacle states."
        )

        return self.inner()

    def step(self) -> None:
        assert self.time_step is not None, (
            "Time step must be set to advance obstacle states."
        )
        assert self.inner is not None, (
            "Motion predictor must be set to advance obstacle states."
        )

        last = self.history.last()
        x, y = step_obstacles(
            x=last.x_array,
            y=last.y_array,
            velocities=self.velocities,
            time_step=self.time_step,
        )
        self.history = self.inner.history = self.history.append(
            JaxObstacleStatesForTimeStep.create(x=x, y=y, heading=last.heading_array)
        )


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
