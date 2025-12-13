from trajax import NumPyMppi, ControlInputBatch, StateBatch, JaxMppi, types

from numtypes import shape_of
import jax.numpy as jnp
import numpy as np


type NumPyCosts[T: int, M: int] = types.numpy.basic.Costs[T, M]
type JaxCosts[T: int, M: int] = types.jax.basic.Costs[T, M]


class numpy:
    @staticmethod
    def energy() -> NumPyMppi.CostFunction:
        """Cost function that penalizes control energy: cost = sum(u^2)."""

        def energy_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: ControlInputBatch[T, D_u, M],
            states: StateBatch[T, D_x, M],
        ) -> NumPyCosts[T, M]:
            T, M = inputs.time_horizon, inputs.rollout_count

            costs = np.sum(np.asarray(inputs) ** 2, axis=1)

            assert shape_of(costs, matches=(T, M), name="energy costs")

            return types.numpy.basic.costs(array=costs)

        return energy_cost

    @staticmethod
    def quadratic_distance_to_origin() -> NumPyMppi.CostFunction:
        """Cost function that penalizes distance from origin: cost = ||x||^2."""

        def quadratic_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: ControlInputBatch[T, D_u, M],
            states: StateBatch[T, D_x, M],
        ) -> NumPyCosts[T, M]:
            states_array = np.asarray(states)
            return types.numpy.basic.costs(array=np.sum(states_array**2, axis=1))

        return quadratic_cost


class jax:
    @staticmethod
    def energy() -> JaxMppi.CostFunction:
        """Cost function that penalizes control energy: cost = sum(u^2)."""

        def energy_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: JaxMppi.ControlInputBatch[T, D_u, M],
            states: JaxMppi.StateBatch[T, D_x, M],
        ) -> JaxCosts[T, M]:
            return types.jax.basic.costs(array=jnp.sum(inputs.array**2, axis=1))

        return energy_cost

    @staticmethod
    def quadratic_distance_to_origin() -> JaxMppi.CostFunction:
        """Cost function that penalizes distance from origin: cost = ||x||^2."""

        def quadratic_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: JaxMppi.ControlInputBatch[T, D_u, M],
            states: JaxMppi.StateBatch[T, D_x, M],
        ) -> JaxCosts[T, M]:
            return types.jax.basic.costs(array=jnp.sum(states.array**2, axis=1))

        return quadratic_cost
