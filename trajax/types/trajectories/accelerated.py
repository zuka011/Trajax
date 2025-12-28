from typing import cast, overload
from dataclasses import dataclass

from trajax.types.array import jaxtyped
from trajax.types.trajectories.common import (
    D_r,
    D_R,
    PathParameters,
    ReferencePoints,
)

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dims, D

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class JaxPathParameters[T: int, M: int](PathParameters[T, M]):
    array: Float[JaxArray, "T M"]

    @overload
    @staticmethod
    def create[T_: int, M_: int](
        array: Array[Dims[T_, M_]],
    ) -> "JaxPathParameters[T_, M_]":
        """Creates a JAX path parameters instance from a NumPy array."""
        ...

    @overload
    @staticmethod
    def create[T_: int, M_: int](
        array: Float[JaxArray, "T M"],
        *,
        horizon: T_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "JaxPathParameters[T_, M_]":
        """Creates a JAX path parameters instance from a JAX array."""
        ...

    @staticmethod
    def create[T_: int, M_: int](
        array: Array[Dims[T_, M_]] | Float[JaxArray, "T M"],
        *,
        horizon: T_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "JaxPathParameters[T_, M_]":
        return JaxPathParameters(array=jnp.asarray(array))

    def __array__(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array)

    @property
    def horizon(self) -> T:
        return cast(T, self.array.shape[0])

    @property
    def rollout_count(self) -> M:
        return cast(M, self.array.shape[1])


@jaxtyped
@dataclass(frozen=True)
class JaxPositions[T: int, M: int]:
    x: Float[JaxArray, "T M"]
    y: Float[JaxArray, "T M"]

    @staticmethod
    def create[T_: int, M_: int](
        *,
        x: Float[JaxArray, "T M"],
        y: Float[JaxArray, "T M"],
        horizon: T_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "JaxPositions[T_, M_]":
        """Creates a JAX positions instance from x and y coordinate arrays."""
        horizon = horizon if horizon is not None else cast(T_, x.shape[0])
        rollout_count = (
            rollout_count if rollout_count is not None else cast(M_, x.shape[1])
        )

        assert x.shape == y.shape == (horizon, rollout_count), (
            f"Expected x and y to have shape {(horizon, rollout_count)}, "
            f"but got {x.shape} and {y.shape}."
        )

        return JaxPositions(x=x, y=y)

    def __array__(self) -> Array[Dims[T, D[2], M]]:
        return np.asarray(jnp.stack([self.x, self.y], axis=-1).transpose(0, 2, 1))


@jaxtyped
@dataclass(frozen=True)
class JaxHeadings[T: int, M: int]:
    theta: Float[JaxArray, "T M"]

    @staticmethod
    def create[T_: int, M_: int](
        *,
        theta: Float[JaxArray, "T M"],
        horizon: T_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "JaxHeadings[T_, M_]":
        """Creates a JAX headings instance from an array of headings."""
        return JaxHeadings(theta=theta)


@dataclass(frozen=True)
class JaxReferencePoints[T: int, M: int](ReferencePoints[T, M]):
    array: Float[JaxArray, f"T {D_R} M"]

    @overload
    @staticmethod
    def create[T_: int, M_: int](
        *,
        x: Array[Dims[T_, M_]],
        y: Array[Dims[T_, M_]],
        heading: Array[Dims[T_, M_]],
    ) -> "JaxReferencePoints[T_, M_]":
        """Creates a JAX reference points instance from NumPy arrays."""
        ...

    @overload
    @staticmethod
    def create[T_: int, M_: int](
        *,
        x: Float[JaxArray, "T M"],
        y: Float[JaxArray, "T M"],
        heading: Float[JaxArray, "T M"],
        horizon: T_,
        rollout_count: M_,
    ) -> "JaxReferencePoints[T_, M_]":
        """Creates a JAX reference points instance from JAX arrays."""
        ...

    @staticmethod
    def create[T_: int, M_: int](
        *,
        x: Array[Dims[T_, M_]] | Float[JaxArray, "T M"],
        y: Array[Dims[T_, M_]] | Float[JaxArray, "T M"],
        heading: Array[Dims[T_, M_]] | Float[JaxArray, "T M"],
        horizon: T_ | None = None,
        rollout_count: M_ | None = None,
    ) -> "JaxReferencePoints[T_, M_]":
        horizon = horizon if horizon is not None else cast(T_, x.shape[0])
        rollout_count = (
            rollout_count if rollout_count is not None else cast(M_, x.shape[1])
        )

        assert x.shape == y.shape == heading.shape == (horizon, rollout_count), (
            f"Expected x, y, and heading to have shape {(horizon, rollout_count)}, "
            f"but got {x.shape}, {y.shape}, and {heading.shape}."
        )

        return JaxReferencePoints(
            array=stack(
                x=jnp.asarray(x), y=jnp.asarray(y), heading=jnp.asarray(heading)
            )
        )

    def __array__(self) -> Array[Dims[T, D_r, M]]:
        return np.asarray(self.array)

    def x(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array[:, 0])

    def y(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array[:, 1])

    def heading(self) -> Array[Dims[T, M]]:
        return np.asarray(self.array[:, 2])

    @property
    def x_array(self) -> Float[JaxArray, "T M"]:
        return self.array[:, 0]

    @property
    def y_array(self) -> Float[JaxArray, "T M"]:
        return self.array[:, 1]

    @property
    def heading_array(self) -> Float[JaxArray, "T M"]:
        return self.array[:, 2]


@jax.jit
@jaxtyped
def stack(
    *,
    x: Float[JaxArray, "T M"],
    y: Float[JaxArray, "T M"],
    heading: Float[JaxArray, "T M"],
) -> Float[JaxArray, f"T {D_R} M"]:
    return jnp.stack([x, y, heading], axis=-1).transpose(0, 2, 1)
