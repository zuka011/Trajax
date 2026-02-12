from typing import Final
from dataclasses import dataclass

from trajax import types

from numtypes import Array
from jaxtyping import Array as JaxArray

import tests.dsl.mppi as data

import jax.numpy as jnp
import numpy as np


@dataclass(kw_only=True, frozen=True)
class NumPySimplePredictionCreator:
    resize_states_to: int | None = None

    def __call__(
        self,
        *,
        states: types.numpy.ObstacleStateSequences,
        covariances: types.numpy.Covariance,
    ) -> types.numpy.simple.ObstacleStates:
        return data.numpy.simple_obstacle_states(
            states=self.pad(states), covariance=covariances
        )

    def empty(self, *, horizon: int) -> types.numpy.simple.ObstacleStates:
        return data.numpy.simple_obstacle_states(
            states=np.empty((horizon, 0, 0)), covariance=np.empty((horizon, 0, 0))
        )

    def pad(self, states: types.numpy.ObstacleStateSequences) -> Array:
        if self.resize_states_to is None:
            return states.array

        if (dimension := states.dimension) >= self.resize_states_to:
            return states.array[:, : self.resize_states_to, :]

        padded_array = np.zeros((states.horizon, self.resize_states_to, states.count))
        padded_array[:, :dimension, :] = states.array

        return padded_array


class NumPyIntegratorPredictionCreator:
    def __call__(
        self,
        *,
        states: types.numpy.integrator.ObstacleStateSequences,
        covariances: types.numpy.PoseCovariance,
    ) -> types.numpy.Obstacle2dPoses:
        return data.numpy.obstacle_2d_poses(
            x=states.array[:, 0, :],
            y=states.array[:, 1, :],
            heading=states.array[:, 2, :],
            covariance=covariances,
        )

    def empty(self, *, horizon: int) -> types.numpy.Obstacle2dPoses:
        return data.numpy.obstacle_2d_poses(
            x=np.empty((horizon, 0)),
            y=np.empty((horizon, 0)),
            heading=np.empty((horizon, 0)),
        )


class NumPyBicyclePredictionCreator:
    def __call__(
        self,
        *,
        states: types.numpy.bicycle.ObstacleStateSequences,
        covariances: types.numpy.PoseCovariance | None,
    ) -> types.numpy.Obstacle2dPoses:
        return data.numpy.obstacle_2d_poses(
            x=states.x(),
            y=states.y(),
            heading=states.heading(),
            covariance=covariances,
        )

    def empty(self, *, horizon: int) -> types.numpy.Obstacle2dPoses:
        return data.numpy.obstacle_2d_poses(
            x=np.empty((horizon, 0)),
            y=np.empty((horizon, 0)),
            heading=np.empty((horizon, 0)),
        )


class NumPyUnicyclePredictionCreator:
    def __call__(
        self,
        *,
        states: types.numpy.unicycle.ObstacleStateSequences,
        covariances: types.numpy.PoseCovariance | None,
    ) -> types.numpy.Obstacle2dPoses:
        return data.numpy.obstacle_2d_poses(
            x=states.x(),
            y=states.y(),
            heading=states.heading(),
            covariance=covariances,
        )

    def empty(self, *, horizon: int) -> types.numpy.Obstacle2dPoses:
        return data.numpy.obstacle_2d_poses(
            x=np.empty((horizon, 0)),
            y=np.empty((horizon, 0)),
            heading=np.empty((horizon, 0)),
        )


@dataclass(kw_only=True, frozen=True)
class JaxSimplePredictionCreator:
    resize_states_to: int | None = None

    def __call__(
        self,
        *,
        states: types.jax.ObstacleStateSequences,
        covariances: types.jax.Covariance,
    ) -> types.jax.simple.ObstacleStates:
        return data.jax.simple_obstacle_states(
            states=self.pad(states), covariance=covariances
        )

    def empty(self, *, horizon: int) -> types.jax.simple.ObstacleStates:
        return data.jax.simple_obstacle_states(
            states=jnp.empty((horizon, 0, 0)), covariance=jnp.empty((horizon, 0, 0))
        )

    def pad(self, states: types.jax.ObstacleStateSequences) -> JaxArray:
        if self.resize_states_to is None:
            return states.array

        if (dimension := states.dimension) >= self.resize_states_to:
            return states.array[:, : self.resize_states_to, :]

        padded_array = jnp.zeros((states.horizon, self.resize_states_to, states.count))
        padded_array = padded_array.at[:, :dimension, :].set(states.array)

        return padded_array


class JaxIntegratorPredictionCreator:
    def __call__(
        self,
        *,
        states: types.jax.integrator.ObstacleStateSequences,
        covariances: types.jax.PoseCovariance,
    ) -> types.jax.Obstacle2dPoses:
        return data.jax.obstacle_2d_poses(
            x=states.array[:, 0, :],
            y=states.array[:, 1, :],
            heading=states.array[:, 2, :],
            covariance=covariances,
        )

    def empty(self, *, horizon: int) -> types.jax.Obstacle2dPoses:
        return data.jax.obstacle_2d_poses(
            x=jnp.empty((horizon, 0)),
            y=jnp.empty((horizon, 0)),
            heading=jnp.empty((horizon, 0)),
        )


class JaxBicyclePredictionCreator:
    def __call__(
        self,
        *,
        states: types.jax.bicycle.ObstacleStateSequences,
        covariances: types.jax.PoseCovariance,
    ) -> types.jax.Obstacle2dPoses:
        return data.jax.obstacle_2d_poses(
            x=states.x_array,
            y=states.y_array,
            heading=states.heading_array,
            covariance=covariances,
        )

    def empty(self, *, horizon: int) -> types.jax.Obstacle2dPoses:
        return data.jax.obstacle_2d_poses(
            x=jnp.empty((horizon, 0)),
            y=jnp.empty((horizon, 0)),
            heading=jnp.empty((horizon, 0)),
        )


class JaxUnicyclePredictionCreator:
    def __call__(
        self,
        *,
        states: types.jax.unicycle.ObstacleStateSequences,
        covariances: types.jax.PoseCovariance,
    ) -> types.jax.Obstacle2dPoses:
        return data.jax.obstacle_2d_poses(
            x=states.x_array,
            y=states.y_array,
            heading=states.heading_array,
            covariance=covariances,
        )

    def empty(self, *, horizon: int) -> types.jax.Obstacle2dPoses:
        return data.jax.obstacle_2d_poses(
            x=jnp.empty((horizon, 0)),
            y=jnp.empty((horizon, 0)),
            heading=jnp.empty((horizon, 0)),
        )


class prediction_creator:
    class numpy:
        simple: Final = NumPySimplePredictionCreator
        integrator: Final = NumPyIntegratorPredictionCreator
        bicycle: Final = NumPyBicyclePredictionCreator
        unicycle: Final = NumPyUnicyclePredictionCreator

    class jax:
        simple: Final = JaxSimplePredictionCreator
        integrator: Final = JaxIntegratorPredictionCreator
        bicycle: Final = JaxBicyclePredictionCreator
        unicycle: Final = JaxUnicyclePredictionCreator
