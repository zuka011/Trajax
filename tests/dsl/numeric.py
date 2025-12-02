from typing import cast
from dataclasses import dataclass

from numtypes import Array, Shape, Dims, shape_of
import numpy as np


@dataclass(frozen=True)
class DisplacementEstimates[ShapeT: Shape]:
    delta_x: Array[ShapeT]
    delta_y: Array[ShapeT]


class estimate:
    @staticmethod
    def displacements[T: int, M: int](
        *,
        velocities: Array[Dims[T, M]],
        orientations: Array[Dims[T, M]],
        time_step_size: float,
    ) -> DisplacementEstimates[Dims[int, M]]:
        """
        Estimate displacement using the trapezoidal rule.

        Given velocity magnitudes and orientations at each timestep, computes
        the expected displacement between consecutive timesteps using the
        trapezoidal approximation: displacement ≈ (v_start + v_end) / 2 * dt.
        """
        T, M = velocities.shape
        vx = velocities * np.cos(orientations)
        vy = velocities * np.sin(orientations)

        delta_x = (vx[:-1] + vx[1:]) / 2 * time_step_size
        delta_y = (vy[:-1] + vy[1:]) / 2 * time_step_size

        assert shape_of(delta_x, matches=(T - 1, M), name="estimated delta_x")
        assert shape_of(delta_y, matches=(T - 1, M), name="estimated delta_y")

        return DisplacementEstimates(delta_x=delta_x, delta_y=delta_y)


class compute:
    @staticmethod
    def angular_distance[ShapeT: Shape](
        theta_1: float | Array[ShapeT], theta_2: float | Array[ShapeT]
    ) -> float | Array[ShapeT]:
        """
        Computes the shortest magnitude distance between two angles.
        Always returns a positive value in [0, π].

        Example:
            angular_distance(0.1, 6.18) -> 0.1
        """
        theta_1 = cast(Array[ShapeT], np.asarray(theta_1))
        theta_2 = cast(Array[ShapeT], np.asarray(theta_2))

        diff = np.abs(theta_1 - theta_2) % (2 * np.pi)
        wrapped = np.minimum(diff, 2 * np.pi - diff)

        assert shape_of(wrapped, matches=theta_1.shape, name="angular distance")

        return wrapped
