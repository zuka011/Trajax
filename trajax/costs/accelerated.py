from typing import Protocol, overload, cast
from dataclasses import dataclass

from trajax.type import jaxtyped
from trajax.types import types
from trajax.trajectory import Trajectory
from trajax.mppi import JaxMppi

import jax
import jax.numpy as jnp
from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims


type PathParameters[T: int = int, M: int = int] = types.jax.PathParameters[T, M]
type Positions[T: int = int, M: int = int] = types.jax.Positions[T, M]
type ReferencePoints[T: int = int, M: int = int] = types.jax.ReferencePoints[T, M]


class PathParameterExtractor[StateT: JaxMppi.StateBatch](Protocol):
    def __call__(self, states: StateT, /) -> PathParameters:
        """Extracts path parameters from a batch of states."""
        ...


class PathVelocityExtractor[InputT: JaxMppi.ControlInputBatch](Protocol):
    def __call__(self, inputs: InputT, /) -> JaxArray:
        """Extracts path velocities from a batch of control inputs."""
        ...


class PositionExtractor[StateT: JaxMppi.StateBatch](Protocol):
    def __call__(self, states: StateT, /) -> Positions:
        """Extracts (x, y) positions from a batch of states."""
        ...


@dataclass(kw_only=True, frozen=True)
class ContouringCost[StateT: JaxMppi.StateBatch]:
    reference: Trajectory[PathParameters, ReferencePoints]
    path_parameter_extractor: PathParameterExtractor[StateT]
    position_extractor: PositionExtractor[StateT]
    weight: float

    @staticmethod
    def create[S: JaxMppi.StateBatch](
        *,
        reference: Trajectory[PathParameters, ReferencePoints],
        path_parameter_extractor: PathParameterExtractor[S],
        position_extractor: PositionExtractor[S],
        weight: float,
    ) -> "ContouringCost[S]":
        """Creates a contouring cost implemented with JAX.

        Args:
            reference: The reference trajectory to follow.
            path_parameter_extractor: Extracts the path parameters from a state batch.
            position_extractor: Extracts the (x, y) positions from a state batch.
            weight: The weight of the contouring cost.
        """
        return ContouringCost(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: JaxMppi.ControlInputBatch[T, int, M], states: StateT
    ) -> JaxMppi.Costs[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading_array
        positions = self.position_extractor(states)

        return types.jax.basic.costs(
            contour_cost(
                heading=heading,
                x=positions.x,
                y=positions.y,
                x_ref=ref_points.x_array,
                y_ref=ref_points.y_array,
                weight=self.weight,
            )
        )


@dataclass(kw_only=True, frozen=True)
class LagCost[StateT: JaxMppi.StateBatch]:
    reference: Trajectory[PathParameters, ReferencePoints]
    path_parameter_extractor: PathParameterExtractor[StateT]
    position_extractor: PositionExtractor[StateT]
    weight: float

    @staticmethod
    def create[S: JaxMppi.StateBatch](
        *,
        reference: Trajectory[PathParameters, ReferencePoints],
        path_parameter_extractor: PathParameterExtractor[S],
        position_extractor: PositionExtractor[S],
        weight: float,
    ) -> "LagCost[S]":
        """Creates a lag cost implemented with JAX.

        Args:
            reference: The reference trajectory to follow.
            path_parameter_extractor: Extracts the path parameters from a state batch.
            position_extractor: Extracts the (x, y) positions from a state batch.
            weight: The weight of the lag cost.
        """
        return LagCost(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: JaxMppi.ControlInputBatch[T, int, M], states: StateT
    ) -> JaxMppi.Costs[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading_array
        positions = self.position_extractor(states)

        return types.jax.basic.costs(
            lag_cost(
                heading=heading,
                x=positions.x,
                y=positions.y,
                x_ref=ref_points.x_array,
                y_ref=ref_points.y_array,
                weight=self.weight,
            )
        )


@dataclass(kw_only=True, frozen=True)
class ProgressCost[InputT: JaxMppi.ControlInputBatch]:
    path_velocity_extractor: PathVelocityExtractor[InputT]
    time_step_size: float
    weight: float

    @staticmethod
    def create[I: JaxMppi.ControlInputBatch](
        *,
        path_velocity_extractor: PathVelocityExtractor[I],
        time_step_size: float,
        weight: float,
    ) -> "ProgressCost[I]":
        """Creates a progress cost implemented with JAX.

        Args:
            path_velocity_extractor: Extracts path velocities from a control input batch.
            time_step_size: The time step size used in the trajectory simulation.
            weight: The weight of the progress cost.
        """
        return ProgressCost(
            path_velocity_extractor=path_velocity_extractor,
            time_step_size=time_step_size,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: InputT, states: JaxMppi.StateBatch[T, int, M]
    ) -> JaxMppi.Costs[T, M]:
        path_velocities = self.path_velocity_extractor(inputs)

        return types.jax.basic.costs(
            progress_cost(
                path_velocities=path_velocities,
                time_step_size=self.time_step_size,
                weight=self.weight,
            )
        )


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class ControlSmoothingCost[D_u: int]:
    weights: Float[JaxArray, "D_u"]
    dimensions: D_u

    @overload
    @staticmethod
    def create[D_u_: int](
        *,
        weights: Array[Dims[D_u_]],
    ) -> "ControlSmoothingCost[D_u_]":
        """Creates a control smoothing cost implemented with JAX.

        Args:
            weights: The weights for each control input dimension.
        """
        ...

    @overload
    @staticmethod
    def create[D_u_: int](
        *,
        weights: Float[JaxArray, "D_u_"],
        dimensions: D_u_,
    ) -> "ControlSmoothingCost[D_u_]":
        """Creates a control smoothing cost implemented with JAX.

        Args:
            weights: The weights for each control input dimension.
            dimensions: The number of control input dimensions.
        """
        ...

    @staticmethod
    def create[D_u_: int](
        *,
        weights: Array[Dims[D_u_]] | Float[JaxArray, "D_u_"],
        dimensions: D_u_ | None = None,
    ) -> "ControlSmoothingCost[D_u_]":
        dimensions = (
            dimensions if dimensions is not None else cast(D_u_, weights.shape[0])
        )
        return ControlSmoothingCost(weights=jnp.asarray(weights), dimensions=dimensions)

    def __call__[T: int, M: int](
        self,
        *,
        inputs: JaxMppi.ControlInputBatch[T, D_u, M],
        states: JaxMppi.StateBatch[T, int, M],
    ) -> JaxMppi.Costs[T, M]:
        return types.jax.basic.costs(
            control_smoothing_cost(inputs=inputs.array, weights=self.weights)
        )


@jax.jit
@jaxtyped
def contour_cost(
    *,
    heading: Float[JaxArray, "T M"],
    x: Float[JaxArray, "T M"],
    y: Float[JaxArray, "T M"],
    x_ref: Float[JaxArray, "T M"],
    y_ref: Float[JaxArray, "T M"],
    weight: Scalar,
) -> Float[JaxArray, "T M"]:
    error = jnp.sin(heading) * (x - x_ref) - jnp.cos(heading) * (y - y_ref)
    return weight * error**2


@jax.jit
@jaxtyped
def lag_cost(
    *,
    heading: Float[JaxArray, "T M"],
    x: Float[JaxArray, "T M"],
    y: Float[JaxArray, "T M"],
    x_ref: Float[JaxArray, "T M"],
    y_ref: Float[JaxArray, "T M"],
    weight: Scalar,
) -> Float[JaxArray, "T M"]:
    # TODO: Add test to make sure error has correct sign.
    error = jnp.cos(heading) * (x - x_ref) + jnp.sin(heading) * (y - y_ref)
    return weight * error**2


@jax.jit
@jaxtyped
def progress_cost(
    *, path_velocities: Float[JaxArray, "T M"], time_step_size: Scalar, weight: Scalar
) -> Float[JaxArray, "T M"]:
    return -weight * path_velocities * time_step_size


@jax.jit
@jaxtyped
def control_smoothing_cost(
    *, inputs: Float[JaxArray, "T D_u M"], weights: Float[JaxArray, " D_u"]
) -> Float[JaxArray, "T M"]:
    diffs = jnp.diff(inputs, axis=0, prepend=inputs[0:1, :, :])
    squared_diffs = diffs**2
    weighted_squared_diffs = squared_diffs * weights[jnp.newaxis, :, jnp.newaxis]
    cost_per_time_step = jnp.sum(weighted_squared_diffs, axis=1)
    return cost_per_time_step
