from typing import Protocol
from dataclasses import dataclass

from trajax.type import jaxtyped
from trajax.types import types
from trajax.trajectory import Trajectory
from trajax.mppi import JaxMppi

import jax
import jax.numpy as jnp
from jaxtyping import Array as JaxArray, Float, Scalar


type PathParameters[T: int = int, M: int = int] = types.jax.PathParameters[T, M]
type Positions[T: int = int, M: int = int] = types.jax.Positions[T, M]
type ReferencePoints[T: int = int, M: int = int] = types.jax.ReferencePoints[T, M]


class ParameterExtractor[D_x: int](Protocol):
    def __call__[T: int, M: int](
        self, states: JaxMppi.StateBatch[T, D_x, M]
    ) -> PathParameters[T, M]:
        """Extracts path parameters from a batch of states."""
        ...


class PositionExtractor[D_x: int](Protocol):
    def __call__[T: int, M: int](
        self, states: JaxMppi.StateBatch[T, D_x, M]
    ) -> Positions[T, M]:
        """Extracts (x, y) positions from a batch of states."""
        ...


@dataclass(kw_only=True, frozen=True)
class ContouringCost[D_x: int]:
    reference: Trajectory[PathParameters, ReferencePoints]
    path_parameter_extractor: ParameterExtractor[D_x]
    position_extractor: PositionExtractor[D_x]
    weight: float

    @staticmethod
    def create[D_x_: int = int](
        *,
        reference: Trajectory[PathParameters, ReferencePoints],
        path_parameter_extractor: ParameterExtractor[D_x_],
        position_extractor: PositionExtractor[D_x_],
        weight: float,
    ) -> "ContouringCost[D_x_]":
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
        self,
        *,
        inputs: JaxMppi.ControlInputBatch[T, int, M],
        states: JaxMppi.StateBatch[T, D_x, M],
    ) -> JaxMppi.Costs[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading
        positions = self.position_extractor(states)

        return types.jax.basic.costs(
            contour_cost(
                heading=heading,
                x=positions.x,
                y=positions.y,
                x_ref=ref_points.x,
                y_ref=ref_points.y,
                weight=self.weight,
            )
        )


@dataclass(kw_only=True, frozen=True)
class LagCost[D_x: int]:
    reference: Trajectory[PathParameters, ReferencePoints]
    path_parameter_extractor: ParameterExtractor[D_x]
    position_extractor: PositionExtractor[D_x]
    weight: float

    @staticmethod
    def create[D_x_: int = int](
        *,
        reference: Trajectory[PathParameters, ReferencePoints],
        path_parameter_extractor: ParameterExtractor[D_x_],
        position_extractor: PositionExtractor[D_x_],
        weight: float,
    ) -> "LagCost[D_x_]":
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
        self,
        *,
        inputs: JaxMppi.ControlInputBatch[T, int, M],
        states: JaxMppi.StateBatch[T, D_x, M],
    ) -> JaxMppi.Costs[T, M]:
        ref_points = self.reference.query(self.path_parameter_extractor(states))
        heading = ref_points.heading
        positions = self.position_extractor(states)

        return types.jax.basic.costs(
            lag_cost(
                heading=heading,
                x=positions.x,
                y=positions.y,
                x_ref=ref_points.x,
                y_ref=ref_points.y,
                weight=self.weight,
            )
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
