from typing import Self
from dataclasses import dataclass

from trajax.types import NumPyObstacleSimulator
from trajax.obstacles.basic import NumPyObstacleStatesForTimeStep

from numtypes import Array, Dims, D, shape_of

import numpy as np


@dataclass(kw_only=True)
class NumPyDynamicObstacleSimulator[K: int](
    NumPyObstacleSimulator[NumPyObstacleStatesForTimeStep[K]]
):
    last: NumPyObstacleStatesForTimeStep[K]
    velocities: Array[Dims[K, D[2]]]

    time_step: float | None = None

    @staticmethod
    def create[K_: int](
        *,
        positions: Array[Dims[K_, D[2]]],
        velocities: Array[Dims[K_, D[2]]],
    ) -> "NumPyDynamicObstacleSimulator[K_]":
        headings = headings_from(velocities)

        return NumPyDynamicObstacleSimulator(
            last=NumPyObstacleStatesForTimeStep.create(
                x=positions[:, 0], y=positions[:, 1], heading=headings
            ),
            velocities=velocities,
        )

    def with_time_step_size(self, time_step_size: float) -> Self:
        return self.__class__(
            last=self.last, velocities=self.velocities, time_step=time_step_size
        )

    def step(self) -> NumPyObstacleStatesForTimeStep[K]:
        assert self.time_step is not None, (
            "Time step must be set to advance obstacle states."
        )

        x, y = step_obstacles(
            x=self.last.x(),
            y=self.last.y(),
            velocities=self.velocities,
            time_step=self.time_step,
        )

        self.last = NumPyObstacleStatesForTimeStep.create(
            x=x, y=y, heading=self.last.heading()
        )

        return self.last

    @property
    def obstacle_count(self) -> int:
        return self.last.count


def step_obstacles[K: int](
    *,
    x: Array[Dims[K]],
    y: Array[Dims[K]],
    velocities: Array[Dims[K, D[2]]],
    time_step: float,
) -> tuple[Array[Dims[K]], Array[Dims[K]]]:
    K = x.shape[0]
    new_x = x + velocities[:, 0] * time_step
    new_y = y + velocities[:, 1] * time_step

    assert shape_of(new_x, matches=(K,))
    assert shape_of(new_y, matches=(K,))

    return new_x, new_y


def headings_from[K: int](velocities: Array[Dims[K, D[2]]]) -> Array[Dims[K]]:
    K = velocities.shape[0]
    speed = np.linalg.norm(velocities, axis=1)
    heading = np.where(
        speed > 1e-6, np.arctan2(velocities[:, 1], velocities[:, 0]), 0.0
    )

    assert shape_of(heading, matches=(K,))

    return heading
