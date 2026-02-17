from typing import Final

from trajax import types

import tests.dsl.mppi as data

import jax.numpy as jnp
import numpy as np


class NumPyIntegratorPredictionCreator:
    def __call__(
        self, *, states: types.numpy.integrator.ObstacleStateSequences
    ) -> types.numpy.Obstacle2dPoses:
        return data.numpy.obstacle_2d_poses(
            x=states.array[:, 0, :],
            y=states.array[:, 1, :],
            heading=states.array[:, 2, :],
            covariance=states.covariance()[:, :3, :3],
        )

    def empty(self, *, horizon: int) -> types.numpy.Obstacle2dPoses:
        return data.numpy.obstacle_2d_poses(
            x=np.empty((horizon, 0)),
            y=np.empty((horizon, 0)),
            heading=np.empty((horizon, 0)),
        )


class NumPyBicyclePredictionCreator:
    def __call__(
        self, *, states: types.numpy.bicycle.ObstacleStateSequences
    ) -> types.numpy.Obstacle2dPoses:
        return states.pose()

    def empty(self, *, horizon: int) -> types.numpy.Obstacle2dPoses:
        return data.numpy.obstacle_2d_poses(
            x=np.empty((horizon, 0)),
            y=np.empty((horizon, 0)),
            heading=np.empty((horizon, 0)),
        )


class NumPyUnicyclePredictionCreator:
    def __call__(
        self, *, states: types.numpy.unicycle.ObstacleStateSequences
    ) -> types.numpy.Obstacle2dPoses:
        return states.pose()

    def empty(self, *, horizon: int) -> types.numpy.Obstacle2dPoses:
        return data.numpy.obstacle_2d_poses(
            x=np.empty((horizon, 0)),
            y=np.empty((horizon, 0)),
            heading=np.empty((horizon, 0)),
        )


class JaxIntegratorPredictionCreator:
    def __call__(
        self, *, states: types.jax.integrator.ObstacleStateSequences
    ) -> types.jax.Obstacle2dPoses:
        return data.jax.obstacle_2d_poses(
            x=states.array[:, 0, :],
            y=states.array[:, 1, :],
            heading=states.array[:, 2, :],
            covariance=states.covariance_array[:, :3, :3],
        )

    def empty(self, *, horizon: int) -> types.jax.Obstacle2dPoses:
        return data.jax.obstacle_2d_poses(
            x=jnp.empty((horizon, 0)),
            y=jnp.empty((horizon, 0)),
            heading=jnp.empty((horizon, 0)),
        )


class JaxBicyclePredictionCreator:
    def __call__(
        self, *, states: types.jax.bicycle.ObstacleStateSequences
    ) -> types.jax.Obstacle2dPoses:
        return states.pose()

    def empty(self, *, horizon: int) -> types.jax.Obstacle2dPoses:
        return data.jax.obstacle_2d_poses(
            x=jnp.empty((horizon, 0)),
            y=jnp.empty((horizon, 0)),
            heading=jnp.empty((horizon, 0)),
        )


class JaxUnicyclePredictionCreator:
    def __call__(
        self, *, states: types.jax.unicycle.ObstacleStateSequences
    ) -> types.jax.Obstacle2dPoses:
        return states.pose()

    def empty(self, *, horizon: int) -> types.jax.Obstacle2dPoses:
        return data.jax.obstacle_2d_poses(
            x=jnp.empty((horizon, 0)),
            y=jnp.empty((horizon, 0)),
            heading=jnp.empty((horizon, 0)),
        )


class prediction_creator:
    class numpy:
        integrator: Final = NumPyIntegratorPredictionCreator
        bicycle: Final = NumPyBicyclePredictionCreator
        unicycle: Final = NumPyUnicyclePredictionCreator

    class jax:
        integrator: Final = JaxIntegratorPredictionCreator
        bicycle: Final = JaxBicyclePredictionCreator
        unicycle: Final = JaxUnicyclePredictionCreator
