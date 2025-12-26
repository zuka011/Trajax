from typing import Protocol, overload, cast
from dataclasses import dataclass

from trajax.type import jaxtyped, DataType
from trajax.mppi import (
    CostFunction,
    JaxControlInputBatch,
    JaxStateBatch,
    JaxCosts,
    ControlInputBatch,
    StateBatch,
)
from trajax.trajectory import (
    Trajectory,
    JaxPathParameters,
    JaxReferencePoints,
    JaxPositions,
    JaxHeadings,
)
from trajax.states import JaxSimpleCosts
from trajax.costs.common import ContouringCost, Error

from jaxtyping import Array as JaxArray, Float, Scalar
from numtypes import Array, Dims

import numpy as np
import jax
import jax.numpy as jnp


class JaxPathParameterExtractor[StateT: JaxStateBatch](Protocol):
    def __call__(self, states: StateT, /) -> JaxPathParameters:
        """Extracts path parameters from a batch of states."""
        ...


class JaxPathVelocityExtractor[InputT: JaxControlInputBatch](Protocol):
    def __call__(self, inputs: InputT, /) -> Float[JaxArray, "T M"]:
        """Extracts path velocities from a batch of control inputs."""
        ...


class JaxPositionExtractor[StateT: JaxStateBatch](Protocol):
    def __call__(self, states: StateT, /) -> JaxPositions:
        """Extracts (x, y) positions from a batch of states."""
        ...


class JaxHeadingExtractor[StateT: JaxStateBatch](Protocol):
    def __call__(self, states: StateT, /) -> JaxHeadings:
        """Extracts heading angles from a batch of states."""
        ...


@jaxtyped
@dataclass(frozen=True)
class JaxError[T: int, M: int](Error[T, M]):
    array: Float[JaxArray, "T M"]

    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, M]]:
        return np.asarray(self.array)


@dataclass(kw_only=True, frozen=True)
class JaxContouringCost[StateT: JaxStateBatch](
    ContouringCost[ControlInputBatch, StateT, JaxCosts]
):
    reference: Trajectory[JaxPathParameters, JaxReferencePoints]
    path_parameter_extractor: JaxPathParameterExtractor[StateT]
    position_extractor: JaxPositionExtractor[StateT]
    weight: float

    @staticmethod
    def create[S: JaxStateBatch](
        *,
        reference: Trajectory[JaxPathParameters, JaxReferencePoints],
        path_parameter_extractor: JaxPathParameterExtractor[S],
        position_extractor: JaxPositionExtractor[S],
        weight: float,
    ) -> "JaxContouringCost[S]":
        """Creates a contouring cost implemented with JAX.

        Args:
            reference: The reference trajectory to follow.
            path_parameter_extractor: Extracts the path parameters from a state batch.
            position_extractor: Extracts the (x, y) positions from a state batch.
            weight: The weight of the contouring cost.
        """
        return JaxContouringCost(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> JaxSimpleCosts[T, M]:
        error = self.error(inputs=inputs, states=states)
        return JaxSimpleCosts(self.weight * error**2)

    def error[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> Float[JaxArray, "T M"]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading_array
        positions = self.position_extractor(states)

        return contour_error(
            heading=heading,
            x=positions.x,
            y=positions.y,
            x_ref=ref_points.x_array,
            y_ref=ref_points.y_array,
        )


@dataclass(kw_only=True, frozen=True)
class JaxLagCost[StateT: JaxStateBatch](
    CostFunction[ControlInputBatch, StateT, JaxCosts]
):
    reference: Trajectory[JaxPathParameters, JaxReferencePoints]
    path_parameter_extractor: JaxPathParameterExtractor[StateT]
    position_extractor: JaxPositionExtractor[StateT]
    weight: float

    @staticmethod
    def create[S: JaxStateBatch](
        *,
        reference: Trajectory[JaxPathParameters, JaxReferencePoints],
        path_parameter_extractor: JaxPathParameterExtractor[S],
        position_extractor: JaxPositionExtractor[S],
        weight: float,
    ) -> "JaxLagCost[S]":
        """Creates a lag cost implemented with JAX.

        Args:
            reference: The reference trajectory to follow.
            path_parameter_extractor: Extracts the path parameters from a state batch.
            position_extractor: Extracts the (x, y) positions from a state batch.
            weight: The weight of the lag cost.
        """
        return JaxLagCost(
            reference=reference,
            path_parameter_extractor=path_parameter_extractor,
            position_extractor=position_extractor,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: ControlInputBatch[T, int, M], states: StateT
    ) -> JaxSimpleCosts[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading_array
        positions = self.position_extractor(states)

        return JaxSimpleCosts(
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
class JaxProgressCost[InputT: JaxControlInputBatch](
    CostFunction[InputT, StateBatch, JaxCosts]
):
    path_velocity_extractor: JaxPathVelocityExtractor[InputT]
    time_step_size: float
    weight: float

    @staticmethod
    def create[I: JaxControlInputBatch](
        *,
        path_velocity_extractor: JaxPathVelocityExtractor[I],
        time_step_size: float,
        weight: float,
    ) -> "JaxProgressCost[I]":
        """Creates a progress cost implemented with JAX.

        Args:
            path_velocity_extractor: Extracts path velocities from a control input batch.
            time_step_size: The time step size used in the trajectory simulation.
            weight: The weight of the progress cost.
        """
        return JaxProgressCost(
            path_velocity_extractor=path_velocity_extractor,
            time_step_size=time_step_size,
            weight=weight,
        )

    def __call__[T: int, M: int](
        self, *, inputs: InputT, states: StateBatch[T, int, M]
    ) -> JaxSimpleCosts[T, M]:
        path_velocities = self.path_velocity_extractor(inputs)

        return JaxSimpleCosts(
            progress_cost(
                path_velocities=path_velocities,
                time_step_size=self.time_step_size,
                weight=self.weight,
            )
        )


@jaxtyped
@dataclass(kw_only=True, frozen=True)
class JaxControlSmoothingCost[D_u: int](
    CostFunction[JaxControlInputBatch[int, D_u, int], StateBatch, JaxCosts]
):
    weights: Float[JaxArray, "D_u"]
    dimensions: D_u

    @overload
    @staticmethod
    def create[D_u_: int](
        *, weights: Array[Dims[D_u_]]
    ) -> "JaxControlSmoothingCost[D_u_]":
        """Creates a control smoothing cost implemented with JAX.

        Args:
            weights: The weights for each control input dimension.
        """
        ...

    @overload
    @staticmethod
    def create[D_u_: int](
        *, weights: Float[JaxArray, "D_u_"], dimensions: D_u_ | None = None
    ) -> "JaxControlSmoothingCost[D_u_]":
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
    ) -> "JaxControlSmoothingCost[D_u_]":
        dimensions = (
            dimensions if dimensions is not None else cast(D_u_, weights.shape[0])
        )
        return JaxControlSmoothingCost(
            weights=jnp.asarray(weights), dimensions=dimensions
        )

    def __call__[T: int, M: int](
        self,
        *,
        inputs: JaxControlInputBatch[T, D_u, M],
        states: StateBatch[T, int, M],
    ) -> JaxSimpleCosts[T, M]:
        return JaxSimpleCosts(
            control_smoothing_cost(inputs=inputs.array, weights=self.weights)
        )


@jax.jit
@jaxtyped
def contour_error(
    *,
    heading: Float[JaxArray, "T M"],
    x: Float[JaxArray, "T M"],
    y: Float[JaxArray, "T M"],
    x_ref: Float[JaxArray, "T M"],
    y_ref: Float[JaxArray, "T M"],
) -> Float[JaxArray, "T M"]:
    return jnp.sin(heading) * (x - x_ref) - jnp.cos(heading) * (y - y_ref)


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
    error = contour_error(heading=heading, x=x, y=y, x_ref=x_ref, y_ref=y_ref)
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
