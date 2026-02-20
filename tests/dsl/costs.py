from faran import types

from numtypes import shape_of

import jax.numpy as jnp
import numpy as np


type NumPyControlInputBatch[T: int, D_u: int, M: int] = types.numpy.ControlInputBatch[
    T, D_u, M
]
type NumPyStateBatch[T: int, D_x: int, M: int] = types.numpy.StateBatch[T, D_x, M]
type NumPyCosts[T: int, M: int] = types.numpy.Costs[T, M]
type NumPyCostFunction[T: int, D_u: int, D_x: int, M: int] = types.numpy.CostFunction[
    T, D_u, D_x, M
]

type JaxControlInputBatch[T: int, D_u: int, M: int] = types.jax.ControlInputBatch[
    T, D_u, M
]
type JaxStateBatch[T: int, D_x: int, M: int] = types.jax.StateBatch[T, D_x, M]
type JaxCosts[T: int, M: int] = types.jax.Costs[T, M]
type JaxCostFunction[T: int, D_u: int, D_x: int, M: int] = types.jax.CostFunction[
    T, D_u, D_x, M
]


class numpy:
    @staticmethod
    def energy() -> NumPyCostFunction:
        """Cost function that penalizes control energy: cost = sum(u^2)."""

        def energy_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: NumPyControlInputBatch[T, D_u, M],
            states: NumPyStateBatch[T, D_x, M],
        ) -> NumPyCosts[T, M]:
            T, M = inputs.horizon, inputs.rollout_count

            costs = np.sum(np.asarray(inputs) ** 2, axis=1)

            assert shape_of(costs, matches=(T, M), name="energy costs")

            return types.numpy.simple.costs(costs)

        return energy_cost

    @staticmethod
    def quadratic_distance_to_origin() -> NumPyCostFunction:
        """Cost function that penalizes distance from origin: cost = ||x||^2."""

        def quadratic_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: NumPyControlInputBatch[T, D_u, M],
            states: NumPyStateBatch[T, D_x, M],
        ) -> NumPyCosts[T, M]:
            states_array = np.asarray(states)
            return types.numpy.simple.costs(np.sum(states_array**2, axis=1))

        return quadratic_cost


class jax:
    @staticmethod
    def energy() -> JaxCostFunction:
        """Cost function that penalizes control energy: cost = sum(u^2)."""

        def energy_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: JaxControlInputBatch[T, D_u, M],
            states: JaxStateBatch[T, D_x, M],
        ) -> JaxCosts[T, M]:
            return types.jax.simple.costs(jnp.sum(inputs.array**2, axis=1))

        return energy_cost

    @staticmethod
    def quadratic_distance_to_origin() -> JaxCostFunction:
        """Cost function that penalizes distance from origin: cost = ||x||^2."""

        def quadratic_cost[T: int, D_u: int, D_x: int, M: int](
            inputs: JaxControlInputBatch[T, D_u, M],
            states: JaxStateBatch[T, D_x, M],
        ) -> JaxCosts[T, M]:
            return types.jax.simple.costs(jnp.sum(states.array**2, axis=1))

        return quadratic_cost
