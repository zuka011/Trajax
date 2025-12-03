from trajax import NumPyMppi, ControlInputBatch, StateBatch, JaxMppi

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dims, shape_of
import jax.numpy as jnp
import numpy as np


class numpy:
    @staticmethod
    def energy() -> NumPyMppi.CostFunction:
        def energy_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: ControlInputBatch[T, D_u, M],
            states: StateBatch[T, D_x, M],
        ) -> Array[Dims[T, M]]:
            T, M = inputs.time_horizon, inputs.rollout_count

            costs = np.sum(np.asarray(inputs) ** 2, axis=1)

            assert shape_of(costs, matches=(T, M), name="energy costs")

            return costs

        return energy_cost


class jax:
    @staticmethod
    def energy() -> JaxMppi.CostFunction:
        def energy_cost(
            *,
            inputs: JaxMppi.ControlInputBatch,
            states: JaxMppi.StateBatch,
        ) -> Float[JaxArray, "T M"]:
            return jnp.sum(inputs.array**2, axis=1)

        return energy_cost
