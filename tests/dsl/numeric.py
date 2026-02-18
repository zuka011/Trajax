from typing import Any, Protocol, cast
from dataclasses import dataclass

from numtypes import Array, Shape, Dims, shape_of

import numpy as np


@dataclass(frozen=True)
class DisplacementEstimates[ShapeT: Shape]:
    delta_x: Array[ShapeT]
    delta_y: Array[ShapeT]


class ArrayConvertible(Protocol):
    def __array__(self, dtype: Any | None = None) -> Array:
        """Returns a representation of this object as a NumPy array."""
        ...


class HasObstacleCount(ArrayConvertible, Protocol):
    @property
    def count(self) -> int:
        """Returns the number of obstacles represented by this object."""
        ...


class ObstacleStatesWrapper[StatesT, D_o: int = Any, K: int = Any](Protocol):
    def __call__(self, states: Array[Dims[D_o, K]]) -> StatesT:
        """Wraps a raw obstacle state array into the type expected by the obstacle model."""
        ...


class ObstacleInputsWrapper[InputsT, D_u: int = Any, K: int = Any](Protocol):
    def __call__(self, inputs: Array[Dims[D_u, K]]) -> InputsT:
        """Wraps a raw control input array into the type expected by the obstacle model."""
        ...


class estimate:
    @staticmethod
    def displacements[T: int, M: int](
        *,
        velocities: Array[Dims[T, M]],
        heading: Array[Dims[T, M]],
        time_step_size: float,
    ) -> DisplacementEstimates[Dims[int, M]]:
        """
        Estimate displacement using the trapezoidal rule.

        Given velocity magnitudes and headings at each timestep, computes
        the expected displacement between consecutive timesteps using the
        trapezoidal approximation: displacement ≈ (v_start + v_end) / 2 * dt.
        """
        T, M = velocities.shape
        vx = velocities * np.cos(heading)
        vy = velocities * np.sin(heading)

        delta_x = (vx[:-1] + vx[1:]) / 2 * time_step_size
        delta_y = (vy[:-1] + vy[1:]) / 2 * time_step_size

        assert shape_of(delta_x, matches=(T - 1, M), name="estimated delta_x")
        assert shape_of(delta_y, matches=(T - 1, M), name="estimated delta_y")

        return DisplacementEstimates(delta_x=delta_x, delta_y=delta_y)


class compute:
    @staticmethod
    def condition_number[D: int = int](
        matrix: Array[Dims[D, D]],
    ) -> float:
        """Computes the condition number (ratio of max to min eigenvalue) of a symmetric matrix."""
        eigenvalues = np.linalg.eigvalsh(matrix)
        return eigenvalues.max() / eigenvalues.min()

    @staticmethod
    def angular_distance[ShapeT: Shape](
        theta_1: float | Array[ShapeT], theta_2: float | Array[ShapeT]
    ) -> float | Array[ShapeT]:
        """
        Computes the shortest magnitude distance between two angles.
        Always returns a positive value in [0, π].

        Example:
            angular_distance(0.1, 2*np.pi) -> 0.1
        """
        theta_1 = cast(Array[ShapeT], np.asarray(theta_1))
        theta_2 = cast(Array[ShapeT], np.asarray(theta_2))

        diff = np.abs(theta_1 - theta_2) % (2 * np.pi)
        wrapped = np.minimum(diff, 2 * np.pi - diff)

        assert shape_of(wrapped, matches=theta_1.shape, name="angular distance")

        return wrapped


class check:
    @staticmethod
    def is_spd[T: int = int, D: int = int, K: int = int](
        matrices: Array[Dims[T, D, D, K]] | Array[Dims[D, D, K]], atol: float = 1e-6
    ) -> bool:
        """Check if matrices are symmetric positive semi-definite."""
        if matrices.ndim == 3:
            matrices = matrices[np.newaxis, ...]

        T, D1, D2, K = matrices.shape

        assert D1 == D2, "Covariance matrices must be square."

        flat = matrices.transpose(0, 3, 1, 2).reshape(-1, D1, D1)

        assert np.abs(flat - flat.swapaxes(-1, -2)).max() <= atol, (
            f"Matrices are not symmetric within tolerance {atol}."
        )

        assert np.linalg.eigvalsh(flat).min() > -atol, (
            f"Matrices are not positive semi-definite within tolerance {atol}."
        )

        return True

    @staticmethod
    def has_diagonal_padding[T: int = int, D: int = int, K: int = int](
        matrices: Array[Dims[T, D, D, K]],
        *,
        from_dimension: int,
        epsilon: float,
        atol: float = 1e-9,
    ) -> bool:
        """Check that padded region has diagonal values = epsilon and off-diagonals = 0.

        For a matrix padded from D_m to D_p:
            [A B]   A is D_m x D_m (original matrix)
            [C D]   B is D_m x (D_p - D_m) (upper rectangle, should be 0)
                    C is (D_p - D_m) x D_m (lower rectangle, should be 0)
                    D is (D_p - D_m) x (D_p - D_m) (diagonal with epsilon)

        Args:
            matrices: Matrices with shape (T, D, D, K).
            from_dimension: The original (non-padded) dimension.
            epsilon: Expected value on the diagonal of the padded region.
            atol: Absolute tolerance for floating point comparisons.
        """
        D = matrices.shape[1]
        D_m = from_dimension

        if (D_pad := D - D_m) <= 0:
            return True  # NOTE: No padding, nothing to check

        assert np.allclose(
            upper_rectangle := matrices[:, :D_m, D_m:, :], 0, atol=atol
        ), (
            f"Upper rectangle (cross-covariance with padded dimensions) should be 0. Got: {upper_rectangle}"
        )

        assert np.allclose(
            lower_rectangle := matrices[:, D_m:, :D_m, :], 0, atol=atol
        ), (
            f"Lower rectangle (cross-covariance with padded dimensions) should be 0. Got: {lower_rectangle}"
        )

        # NOTE: Transpose so triu/tril apply to matrix dimensions
        padded_region = np.moveaxis(matrices[:, D_m:, D_m:, :], -1, 1)

        assert np.allclose(
            off_diagonals_upper := np.triu(padded_region, k=1), 0, atol=atol
        ) and np.allclose(
            off_diagonals_lower := np.tril(padded_region, k=-1), 0, atol=atol
        ), (
            f"Off-diagonal elements in padded region should be exactly 0. Got upper: {off_diagonals_upper}, lower: {off_diagonals_lower}"
        )

        assert np.allclose(
            diagonals := np.array(
                [matrices[:, D_m + i, D_m + i, :] for i in range(D_pad)]
            ),
            epsilon,
            atol=atol,
        ), f"Diagonal elements in padded region should be {epsilon}. Got: {diagonals}"

        return True
