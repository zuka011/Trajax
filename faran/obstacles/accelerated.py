from typing import Sequence, cast
from dataclasses import dataclass
from functools import cached_property

from faran.types import (
    jaxtyped,
    DataType,
    PoseD_o as D_o,
    POSE_D_O as D_O,
    Device,
    place,
    JaxSampledObstacleStates,
    JaxSampledObstaclePositions,
    JaxSampledObstacleHeadings,
    JaxObstacleStates,
    JaxObstacleStatesForTimeStep,
    JaxObstaclePositions,
    JaxObstaclePositionsForTimeStep,
    JaxObstacleOrientations,
    JaxObstacleOrientationsForTimeStep,
)
from faran.obstacles.basic import NumPyObstacle2dPosesForTimeStep

from numtypes import Array, Dims, D
from jaxtyping import Array as JaxArray, Float

import numpy as np
import jax.numpy as jnp

type ObstacleCovarianceArray[T: int = int, K: int = int] = Float[
    JaxArray, f"T {D_O} {D_O} K"
]


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxSampledObstacle2dPoses[T: int, K: int, N: int](
    JaxSampledObstacleStates[T, D_o, K, N]
):
    """Sampled 2D poses (x, y, heading) with shape (T, POSE_D_O, K, N)."""

    _x: Float[JaxArray, "T K N"]
    _y: Float[JaxArray, "T K N"]
    _heading: Float[JaxArray, "T K N"]

    @staticmethod
    def create[T_: int, K_: int, N_: int](
        *,
        x: Array[Dims[T_, K_, N_]] | Float[JaxArray, "T K N"],
        y: Array[Dims[T_, K_, N_]] | Float[JaxArray, "T K N"],
        heading: Array[Dims[T_, K_, N_]] | Float[JaxArray, "T K N"],
    ) -> "JaxSampledObstacle2dPoses[T_, K_, N_]":
        return JaxSampledObstacle2dPoses(
            _x=jnp.asarray(x), _y=jnp.asarray(y), _heading=jnp.asarray(heading)
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K, N]]:
        return self._numpy_array

    def x(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._x)

    def y(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._y)

    def heading(self) -> Array[Dims[T, K, N]]:
        return np.asarray(self._heading)

    def positions(self) -> JaxSampledObstaclePositions[T, K, N]:
        return JaxSampledObstaclePositions.create(x=self._x, y=self._y)

    def headings(self) -> JaxSampledObstacleHeadings[T, K, N]:
        return JaxSampledObstacleHeadings.create(heading=self._heading)

    def at(self, *, time_step: int, sample: int) -> "JaxObstacle2dPosesForTimeStep[K]":
        return JaxObstacle2dPosesForTimeStep.create(
            x=self._x[time_step, :, sample],
            y=self._y[time_step, :, sample],
            heading=self._heading[time_step, :, sample],
        )

    @property
    def horizon(self) -> T:
        return cast(T, self._x.shape[0])

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[1])

    @property
    def sample_count(self) -> N:
        return cast(N, self._x.shape[2])

    @property
    def x_array(self) -> Float[JaxArray, "T K N"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "T K N"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "T K N"]:
        return self._heading

    @property
    def array(self) -> Float[JaxArray, f"T {D_O} K N"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, f"T {D_O} K N"]:
        return jnp.stack([self._x, self._y, self._heading], axis=1)

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_o, K, N]]:
        return np.asarray(self._array)


@dataclass(kw_only=True, frozen=True)
class JaxObstacle2dPositions[T: int, K: int](JaxObstaclePositions[T, D[2], K]):
    """2D positions (x, y) with shape (T, 2, K)."""

    _x: Float[JaxArray, "T K"]
    _y: Float[JaxArray, "T K"]

    @staticmethod
    def create[T_: int, K_: int](
        *,
        x: Float[JaxArray, "T K"],
        y: Float[JaxArray, "T K"],
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacle2dPositions[T_, K_]":
        return JaxObstacle2dPositions(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[2], K]]:
        return self._numpy_array

    @property
    def horizon(self) -> T:
        return cast(T, self._x.shape[0])

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[1])

    @property
    def array(self) -> Float[JaxArray, "T 2 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, "T 2 K"]:
        return jnp.stack([self._x, self._y], axis=1)

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D[2], K]]:
        return np.asarray(self._array)


@dataclass(kw_only=True, frozen=True)
class JaxObstacle2dPositionsForTimeStep[K: int](
    JaxObstaclePositionsForTimeStep[D[2], K]
):
    """2D positions (x, y) for a single time step with shape (2, K)."""

    _x: Float[JaxArray, "K"]
    _y: Float[JaxArray, "K"]

    @staticmethod
    def create[K_: int](
        *,
        x: Float[JaxArray, "K"],
        y: Float[JaxArray, "K"],
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacle2dPositionsForTimeStep[K_]":
        return JaxObstacle2dPositionsForTimeStep(_x=x, _y=y)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D[2], K]]:
        return self._numpy_array

    @property
    def dimension(self) -> D[2]:
        return 2

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[0])

    @property
    def array(self) -> Float[JaxArray, "2 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, "2 K"]:
        return jnp.stack([self._x, self._y], axis=0)

    @cached_property
    def _numpy_array(self) -> Array[Dims[D[2], K]]:
        return np.asarray(self._array)


@dataclass(kw_only=True, frozen=True)
class JaxObstacleHeadings[T: int, K: int](JaxObstacleOrientations[T, D[1], K]):
    """Obstacle headings with shape (T, 1, K)."""

    _heading: Float[JaxArray, "T K"]

    @staticmethod
    def create[T_: int, K_: int](
        *,
        heading: Float[JaxArray, "T K"],
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacleHeadings[T_, K_]":
        return JaxObstacleHeadings(_heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D[1], K]]:
        return self._numpy_array

    @property
    def horizon(self) -> T:
        return cast(T, self._heading.shape[0])

    @property
    def dimension(self) -> D[1]:
        return 1

    @property
    def count(self) -> K:
        return cast(K, self._heading.shape[1])

    @property
    def array(self) -> Float[JaxArray, "T 1 K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, "T 1 K"]:
        return self._heading[:, jnp.newaxis, :]

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D[1], K]]:
        return np.asarray(self._array)


@dataclass(kw_only=True, frozen=True)
class JaxObstacleHeadingsForTimeStep[K: int](
    JaxObstacleOrientationsForTimeStep[D[1], K]
):
    """Obstacle headings for a single time step with shape (1, K)."""

    _heading: Float[JaxArray, "K"]

    @staticmethod
    def create[K_: int](
        *, heading: Float[JaxArray, "K"], obstacle_count: K_ | None = None
    ) -> "JaxObstacleHeadingsForTimeStep[K_]":
        return JaxObstacleHeadingsForTimeStep(_heading=heading)

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D[1], K]]:
        return self._numpy_array

    @property
    def dimension(self) -> D[1]:
        return 1

    @property
    def count(self) -> K:
        return cast(K, self._heading.shape[0])

    @property
    def array(self) -> Float[JaxArray, "1 K"]:
        return self._jax_array

    @cached_property
    def _jax_array(self) -> Float[JaxArray, "1 K"]:
        return self._heading[jnp.newaxis, :]

    @cached_property
    def _numpy_array(self) -> Array[Dims[D[1], K]]:
        return np.asarray(self._jax_array)


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxObstacle2dPoses[T: int, K: int](
    JaxObstacleStates[
        T,
        D_o,
        K,
        JaxSampledObstacle2dPoses[T, K, D[1]],
        "JaxObstacle2dPosesForTimeStep[K]",
    ]
):
    """2D poses (x, y, heading) with shape (T, POSE_D_O, K)."""

    _x: Float[JaxArray, "T K"]
    _y: Float[JaxArray, "T K"]
    _heading: Float[JaxArray, "T K"]
    _covariance: ObstacleCovarianceArray[T, K] | None = None

    @staticmethod
    def empty[T_: int, K_: int = D[0]](
        *, horizon: T_, obstacle_count: K_ = 0
    ) -> "JaxObstacle2dPoses[T_, K_]":
        """Creates obstacle states for zero obstacles over the given time horizon."""
        empty = jnp.full((horizon, obstacle_count), fill_value=jnp.nan)

        return JaxObstacle2dPoses.create(x=empty, y=empty, heading=empty)

    @staticmethod
    def sampled[N: int](  # type: ignore
        *,
        x: Array[Dims[T, K, N]] | Float[JaxArray, "T K N"],
        y: Array[Dims[T, K, N]] | Float[JaxArray, "T K N"],
        heading: Array[Dims[T, K, N]] | Float[JaxArray, "T K N"],
    ) -> JaxSampledObstacle2dPoses[T, K, N]:
        return JaxSampledObstacle2dPoses.create(
            x=jnp.asarray(x), y=jnp.asarray(y), heading=jnp.asarray(heading)
        )

    @staticmethod
    def wrap[T_: int, K_: int](
        array: Float[JaxArray, f"T {D_O} K"],
        *,
        horizon: T_ | None = None,
        obstacle_count: K_ | None = None,
    ) -> "JaxObstacle2dPoses[T_, K_]":
        horizon = horizon if horizon is not None else cast(T_, array.shape[0])
        obstacle_count = (
            obstacle_count if obstacle_count is not None else cast(K_, array.shape[2])
        )

        assert array.shape == (horizon, D_O, obstacle_count), (
            f"Expected shape (T={horizon}, D_o={D_O}, K={obstacle_count}), "
            f"but got array with shape {array.shape}."
        )

        return JaxObstacle2dPoses.create(
            x=array[:, 0, :], y=array[:, 1, :], heading=array[:, 2, :]
        )

    @staticmethod
    def create[T_: int, K_: int](
        *,
        x: Array[Dims[T_, K_]] | Float[JaxArray, "T K"],
        y: Array[Dims[T_, K_]] | Float[JaxArray, "T K"],
        heading: Array[Dims[T_, K_]] | Float[JaxArray, "T K"],
        covariance: Array[Dims[T_, D_o, D_o, K_]]
        | ObstacleCovarianceArray[T_, K_]
        | None = None,
    ) -> "JaxObstacle2dPoses[T_, K_]":

        return JaxObstacle2dPoses(
            _x=jnp.asarray(x),
            _y=jnp.asarray(y),
            _heading=jnp.asarray(heading),
            _covariance=jnp.asarray(covariance) if covariance is not None else None,
        )

    @staticmethod
    def of_states[T_: int, K_: int](
        obstacle_states: Sequence["JaxObstacle2dPosesForTimeStep[K_]"],
        *,
        horizon: T_ | None = None,
    ) -> "JaxObstacle2dPoses[T_, K_]":
        assert len(obstacle_states) > 0, "Obstacle states sequence must not be empty."
        assert horizon is None or len(obstacle_states) == horizon, (
            f"Expected horizon {horizon}, but got {len(obstacle_states)} obstacle states."
        )

        K = max(states.count for states in obstacle_states)

        def pad(array: Float[JaxArray, "L"]) -> Float[JaxArray, "K"]:
            return jnp.pad(array, (0, K - len(array)), constant_values=jnp.nan)

        x = jnp.stack([pad(states.x_array) for states in obstacle_states])
        y = jnp.stack([pad(states.y_array) for states in obstacle_states])
        heading = jnp.stack([pad(states.heading_array) for states in obstacle_states])

        return JaxObstacle2dPoses.create(x=x, y=y, heading=heading)

    @staticmethod
    def for_time_step[K_: int](
        *,
        x: Array[Dims[K_]] | Float[JaxArray, "K"],
        y: Array[Dims[K_]] | Float[JaxArray, "K"],
        heading: Array[Dims[K_]] | Float[JaxArray, "K"],
        device: Device = "cpu",
    ) -> "JaxObstacle2dPosesForTimeStep[K_]":
        """Creates obstacle states for a single time step.

        Note:
            Since the common case is to further process this data on the CPU first,
            the default device is set to "cpu".
        """
        return JaxObstacle2dPosesForTimeStep.create(
            x=place(x, device=device),
            y=place(y, device=device),
            heading=place(heading, device=device),
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        return self._numpy_array

    def x(self) -> Array[Dims[T, K]]:
        return np.asarray(self._x)

    def y(self) -> Array[Dims[T, K]]:
        return np.asarray(self._y)

    def heading(self) -> Array[Dims[T, K]]:
        return np.asarray(self._heading)

    def covariance(self) -> Array[Dims[T, D_o, D_o, K]] | None:
        return np.asarray(self._covariance) if self._covariance is not None else None

    def positions(self) -> "JaxObstacle2dPositions[T, K]":
        return JaxObstacle2dPositions.create(x=self._x, y=self._y)

    def headings(self) -> "JaxObstacleHeadings[T, K]":
        return JaxObstacleHeadings.create(heading=self._heading)

    def single(self) -> JaxSampledObstacle2dPoses[T, K, D[1]]:
        return JaxSampledObstacle2dPoses.create(
            x=self._x[..., jnp.newaxis],
            y=self._y[..., jnp.newaxis],
            heading=self._heading[..., jnp.newaxis],
        )

    def last(self) -> "JaxObstacle2dPosesForTimeStep[K]":
        return self.at(time_step=self.horizon - 1)

    def at(self, time_step: int) -> "JaxObstacle2dPosesForTimeStep[K]":
        return JaxObstacle2dPosesForTimeStep.create(
            x=self._x[time_step],
            y=self._y[time_step],
            heading=self._heading[time_step],
        )

    @property
    def array(self) -> Float[JaxArray, "T D_o K"]:
        return self._array

    @property
    def x_array(self) -> Float[JaxArray, "T K"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "T K"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "T K"]:
        return self._heading

    @property
    def covariance_array(self) -> ObstacleCovarianceArray[T, K] | None:
        return self._covariance

    @property
    def horizon(self) -> T:
        return cast(T, self._x.shape[0])

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[1])

    @cached_property
    def _array(self) -> Float[JaxArray, "T D_o K"]:
        return jnp.stack([self._x, self._y, self._heading], axis=1)

    @cached_property
    def _numpy_array(self) -> Array[Dims[T, D_o, K]]:
        return np.asarray(self._array)


@dataclass(kw_only=True, frozen=True)
class JaxObstacle2dPosesForTimeStep[K: int](
    JaxObstacleStatesForTimeStep[D_o, K, JaxObstacle2dPoses]
):
    """2D poses (x, y, heading) for a single time step with shape (POSE_D_O, K)."""

    _x: Float[JaxArray, "K"]
    _y: Float[JaxArray, "K"]
    _heading: Float[JaxArray, "K"]

    @staticmethod
    def create[K_: int](
        *,
        x: Array[Dims[K_]] | Float[JaxArray, "K"],
        y: Array[Dims[K_]] | Float[JaxArray, "K"],
        heading: Array[Dims[K_]] | Float[JaxArray, "K"],
    ) -> "JaxObstacle2dPosesForTimeStep[K_]":

        return JaxObstacle2dPosesForTimeStep(
            _x=jnp.asarray(x), _y=jnp.asarray(y), _heading=jnp.asarray(heading)
        )

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_o, K]]:
        return self._numpy_array

    def numpy(self) -> NumPyObstacle2dPosesForTimeStep[K]:
        return NumPyObstacle2dPosesForTimeStep.create(
            x=np.asarray(self._x),
            y=np.asarray(self._y),
            heading=np.asarray(self._heading),
        )

    def x(self) -> Array[Dims[K]]:
        return np.asarray(self._x)

    def y(self) -> Array[Dims[K]]:
        return np.asarray(self._y)

    def heading(self) -> Array[Dims[K]]:
        return np.asarray(self._heading)

    def positions(self) -> JaxObstacle2dPositionsForTimeStep[K]:
        return JaxObstacle2dPositionsForTimeStep.create(x=self._x, y=self._y)

    def headings(self) -> JaxObstacleHeadingsForTimeStep[K]:
        return JaxObstacleHeadingsForTimeStep.create(heading=self._heading)

    def replicate[T: int](self, *, horizon: T) -> JaxObstacle2dPoses[T, K]:
        return JaxObstacle2dPoses.create(
            x=jnp.tile(self._x[jnp.newaxis, :], (horizon, 1)),
            y=jnp.tile(self._y[jnp.newaxis, :], (horizon, 1)),
            heading=jnp.tile(self._heading[jnp.newaxis, :], (horizon, 1)),
        )

    @property
    def dimension(self) -> D_o:
        return D_O

    @property
    def count(self) -> K:
        return cast(K, self._x.shape[0])

    @property
    def x_array(self) -> Float[JaxArray, "K"]:
        return self._x

    @property
    def y_array(self) -> Float[JaxArray, "K"]:
        return self._y

    @property
    def heading_array(self) -> Float[JaxArray, "K"]:
        return self._heading

    @property
    def array(self) -> Float[JaxArray, f"{D_O} K"]:
        return self._array

    @cached_property
    def _array(self) -> Float[JaxArray, f"{D_O} K"]:
        return jnp.stack([self._x, self._y, self._heading], axis=0)

    @cached_property
    def _numpy_array(self) -> Array[Dims[D_o, K]]:
        return np.asarray(self._array)
