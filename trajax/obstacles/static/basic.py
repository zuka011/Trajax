from typing import Self, cast
from dataclasses import dataclass

from trajax.types import NumPyObstacleSimulator
from trajax.obstacles.basic import NumPyObstacleStatesForTimeStep

from numtypes import Array, Dims, D, shape_of

import numpy as np


@dataclass(frozen=True)
class NumPyStaticObstacleSimulator[K: int](
    NumPyObstacleSimulator[NumPyObstacleStatesForTimeStep[K]]
):
    positions: Array[Dims[K, D[2]]]
    headings: Array[Dims[K]]

    @staticmethod
    def empty() -> "NumPyStaticObstacleSimulator[D[0]]":
        positions = np.empty((0, 2))

        assert shape_of(positions, matches=(0, 2))

        return NumPyStaticObstacleSimulator.create(positions=positions)

    @staticmethod
    def create[K_: int](
        *, positions: Array[Dims[K_, D[2]]], headings: Array[Dims[K_]] | None = None
    ) -> "NumPyStaticObstacleSimulator[K_]":
        count = positions.shape[0]
        headings = (
            headings
            if headings is not None
            else cast(Array[Dims[K_]], np.zeros(shape=(count,)))
        )

        return NumPyStaticObstacleSimulator(positions=positions, headings=headings)

    def with_time_step_size(self, time_step_size: float) -> Self:
        # Time step does not matter.
        return self

    def step(self) -> NumPyObstacleStatesForTimeStep[K]:
        return NumPyObstacleStatesForTimeStep.create(
            x=self.positions[:, 0],
            y=self.positions[:, 1],
            heading=self.headings,
        )

    @property
    def obstacle_count(self) -> int:
        return self.positions.shape[0]
