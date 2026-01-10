from dataclasses import dataclass
from functools import cached_property

from trajax.types import (
    Trajectory,
    NumPyPathParameters,
    NumPyReferencePoints,
    NumPyPositions,
    NumPyLateralPositions,
    NumPyLongitudinalPositions,
)

from numtypes import Array, Dims, D, shape_of

import numpy as np


type Vector = Array[Dims[D[2]]]


@dataclass(kw_only=True, frozen=True)
class NumPyLineTrajectory(
    Trajectory[
        NumPyPathParameters,
        NumPyReferencePoints,
        NumPyPositions,
        NumPyLateralPositions,
        NumPyLongitudinalPositions,
    ]
):
    start: Vector
    direction: Vector

    heading: float

    _path_length: float

    @staticmethod
    def create(
        *, start: tuple[float, float], end: tuple[float, float], path_length: float
    ) -> "NumPyLineTrajectory":
        """Generates a straight line trajectory from start to end."""
        return NumPyLineTrajectory(
            start=(start_array := np.array(start)),
            direction=(direction := np.array(end) - start_array),
            heading=np.arctan2(direction[1], direction[0]),
            _path_length=path_length,
        )

    def query[T: int, M: int](
        self, parameters: NumPyPathParameters[T, M]
    ) -> NumPyReferencePoints[T, M]:
        T, M = parameters.horizon, parameters.rollout_count
        normalized = parameters.array / self.path_length

        x, y = (
            self.start[:, np.newaxis, np.newaxis]
            + self.direction[:, np.newaxis, np.newaxis] * normalized
        )
        heading = np.full_like(x, self.heading)

        assert shape_of(x, matches=(T, M), name="x")
        assert shape_of(y, matches=(T, M), name="y")
        assert shape_of(heading, matches=(T, M), name="heading")

        return NumPyReferencePoints.create(x=x, y=y, heading=heading)

    def lateral[T: int, M: int](
        self, positions: NumPyPositions[T, M]
    ) -> NumPyLateralPositions[T, M]:
        T, M = positions.horizon, positions.rollout_count
        relative = positions.array - self.start[:, np.newaxis]

        assert shape_of(relative, matches=(T, 2, M), name="relative")

        lateral = np.einsum("tpm,p->tm", relative, self.perpendicular)

        assert shape_of(lateral, matches=(T, M), name="lateral")

        return NumPyLateralPositions.create(lateral)

    def longitudinal[T: int, M: int](
        self, positions: NumPyPositions[T, M]
    ) -> NumPyLongitudinalPositions[T, M]:
        T, M = positions.horizon, positions.rollout_count
        relative = positions.array - self.start[:, np.newaxis]

        assert shape_of(relative, matches=(T, 2, M), name="relative")

        projection = np.einsum("tpm,p->tm", relative, self.tangent)
        longitudinal = projection * self.path_length / self.line_length

        assert shape_of(longitudinal, matches=(T, M), name="longitudinal")

        return NumPyLongitudinalPositions.create(longitudinal)

    @property
    def path_length(self) -> float:
        return self._path_length

    @cached_property
    def perpendicular(self) -> Vector:
        tangent = self.tangent
        perpendicular = np.array([tangent[1], -tangent[0]])

        assert shape_of(perpendicular, matches=(2,), name="perpendicular")

        return perpendicular

    @cached_property
    def tangent(self) -> Vector:
        tangent = self.direction / self.line_length

        assert shape_of(tangent, matches=(2,), name="tangent")

        return tangent

    @cached_property
    def line_length(self) -> np.floating:
        return np.linalg.norm(self.direction)
