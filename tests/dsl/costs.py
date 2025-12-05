from trajax import NumPyMppi, ControlInputBatch, StateBatch, JaxMppi

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dims, shape_of
import jax.numpy as jnp
import numpy as np


class numpy:
    @staticmethod
    def energy() -> NumPyMppi.CostFunction:
        """Cost function that penalizes control energy: cost = sum(u^2)."""

        def energy_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: ControlInputBatch[T, D_u, M],
            states: StateBatch[T, D_x, M],
        ) -> Array[Dims[T, M]]:
            T, M = inputs.time_horizon, inputs.rollout_count

            costs = np.sum(np.asarray(inputs) ** 2, axis=1)

            assert shape_of(costs, matches=(T, M), name="energy costs")

            return costs

        return energy_cost

    @staticmethod
    def quadratic_distance_to_origin() -> NumPyMppi.CostFunction:
        """Cost function that penalizes distance from origin: cost = ||x||^2."""

        def quadratic_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: ControlInputBatch[T, D_u, M],
            states: StateBatch[T, D_x, M],
        ) -> Array[Dims[T, M]]:
            states_array = np.asarray(states)
            return np.sum(states_array**2, axis=1)

        return quadratic_cost


class jax:
    @staticmethod
    def energy() -> JaxMppi.CostFunction:
        """Cost function that penalizes control energy: cost = sum(u^2)."""

        def energy_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: JaxMppi.ControlInputBatch[T, D_u, M],
            states: JaxMppi.StateBatch[T, D_x, M],
        ) -> Float[JaxArray, "T M"]:
            return jnp.sum(inputs.array**2, axis=1)

        return energy_cost

    @staticmethod
    def quadratic_distance_to_origin() -> JaxMppi.CostFunction:
        """Cost function that penalizes distance from origin: cost = ||x||^2."""

        def quadratic_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: JaxMppi.ControlInputBatch[T, D_u, M],
            states: JaxMppi.StateBatch[T, D_x, M],
        ) -> Float[JaxArray, "T M"]:
            return jnp.sum(states.array**2, axis=1)

        return quadratic_cost
