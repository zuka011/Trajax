from typing import Self
from dataclasses import dataclass

from trajax.types import jaxtyped, JaxObstacleStateProvider, ObstacleMotionPredictor
from trajax.obstacles.accelerated import JaxObstacleStates, JaxObstacleStatesForTimeStep
from trajax.obstacles.history import JaxObstacleStatesRunningHistory

from jaxtyping import Array as JaxArray, Float, Scalar

import jax
import jax.numpy as jnp


@dataclass(kw_only=True)
class JaxDynamicObstacleStateProvider[T: int, K: int](
    JaxObstacleStateProvider[JaxObstacleStates[T, K]]
):
    type MotionPredictor[T_: int, K_: int] = ObstacleMotionPredictor[
        JaxObstacleStates[int, K_], JaxObstacleStates[T_, K_]
    ]

    history: JaxObstacleStatesRunningHistory[int, K]
    velocities: Float[JaxArray, "K 2"]

    horizon: T
    time_step: Scalar | None = None
    predictor: MotionPredictor[T, K] | None = None

    @staticmethod
    def create[T_: int, K_: int](
        *,
        positions: Float[JaxArray, "K 2"],
        velocities: Float[JaxArray, "K 2"],
        horizon: T_,
        obstacle_count: K_ | None = None,
    ) -> "JaxDynamicObstacleStateProvider[T_, K_]":
        headings = headings_from(velocities)

        return JaxDynamicObstacleStateProvider(
            history=JaxObstacleStatesRunningHistory.single(
                JaxObstacleStatesForTimeStep.create(
                    x=positions[:, 0], y=positions[:, 1], heading=headings
                )
            ),
            velocities=velocities,
            horizon=horizon,
        )

    def with_time_step(self, time_step: float) -> Self:
        return self.__class__(
            history=self.history,
            velocities=self.velocities,
            predictor=self.predictor,
            horizon=self.horizon,
            time_step=jnp.asarray(time_step),
        )

    def with_predictor(self, predictor: MotionPredictor) -> Self:
        return self.__class__(
            history=self.history,
            velocities=self.velocities,
            predictor=predictor,
            horizon=self.horizon,
            time_step=self.time_step,
        )

    def __call__(self) -> JaxObstacleStates[T, K]:
        assert self.predictor is not None, (
            "Motion predictor must be set to provide obstacle states."
        )

        return self.predictor.predict(history=self.history.get())

    def step(self) -> None:
        assert self.time_step is not None, (
            "Time step must be set to advance obstacle states."
        )

        last = self.history.last()
        x, y = step_obstacles(
            x=last.x_array,
            y=last.y_array,
            velocities=self.velocities,
            time_step=self.time_step,
        )
        self.history = self.history.append(
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
