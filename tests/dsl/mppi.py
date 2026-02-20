from typing import Final, Sequence

from faran import types

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, NumberArray, Dims

import numpy as np
import jax.numpy as jnp

D_O: Final = types.obstacle.POSE_D_O

type D_o = types.obstacle.PoseD_o
type NumPyState[D_x: int] = types.numpy.simple.State[D_x]
type NumPyStateBatch[T: int, D_x: int, M: int] = types.numpy.simple.StateBatch[
    T, D_x, M
]
type NumPyControlInputSequence[T: int, D_u: int] = (
    types.numpy.simple.ControlInputSequence[T, D_u]
)
type NumPyControlInputBatch[T: int, D_u: int, M: int] = (
    types.numpy.simple.ControlInputBatch[T, D_u, M]
)
type NumPySampledObstacle2dPoses[T: int, K: int, N: int] = (
    types.numpy.SampledObstacle2dPoses[T, K, N]
)
type NumPyObstacleIds[K: int] = types.numpy.ObstacleIds[K]
type NumPyObstacle2dPosesForTimeStep[K: int] = types.numpy.Obstacle2dPosesForTimeStep[K]
type NumPyObstacle2dPoses[T: int, K: int] = types.numpy.Obstacle2dPoses[T, K]
type NumPySimpleObstacleStates[T: int, D_o: int, K: int] = (
    types.numpy.simple.ObstacleStates[T, D_o, K]
)
type NumPyDistance[T: int, V: int, M: int, N: int] = types.numpy.Distance[T, V, M, N]
type NumPyBoundaryDistance[T: int, M: int] = types.numpy.BoundaryDistance[T, M]

type JaxState[D_x: int] = types.jax.simple.State[D_x]
type JaxStateBatch[T: int, D_x: int, M: int] = types.jax.simple.StateBatch[T, D_x, M]
type JaxControlInputSequence[T: int, D_u: int] = types.jax.simple.ControlInputSequence[
    T, D_u
]
type JaxControlInputBatch[T: int, D_u: int, M: int] = (
    types.jax.simple.ControlInputBatch[T, D_u, M]
)
type JaxSampledObstacle2dPoses[T: int, K: int, N: int] = (
    types.jax.SampledObstacle2dPoses[T, K, N]
)
type JaxObstacleIds[K: int] = types.jax.ObstacleIds[K]
type JaxObstacle2dPosesForTimeStep[K: int] = types.jax.Obstacle2dPosesForTimeStep[K]
type JaxObstacle2dPoses[T: int, K: int] = types.jax.Obstacle2dPoses[T, K]
type JaxSimpleObstacleStates[T: int, D_o: int, K: int] = (
    types.jax.simple.ObstacleStates[T, D_o, K]
)
type JaxDistance[T: int, V: int, M: int, N: int] = types.jax.Distance[T, V, M, N]
type JaxBoundaryDistance[T: int, M: int] = types.jax.BoundaryDistance[T, M]


class numpy:
    @staticmethod
    def state[D_x: int](array: Array[Dims[D_x]]) -> NumPyState[D_x]:
        return types.numpy.simple.state(array)

    @staticmethod
    def state_batch[T: int, D_x: int, M: int](
        array: Array[Dims[T, D_x, M]],
    ) -> NumPyStateBatch[T, D_x, M]:
        return types.numpy.simple.state_batch(array)

    @staticmethod
    def control_input_sequence[T: int, D_u: int](
        array: Array[Dims[T, D_u]],
    ) -> NumPyControlInputSequence[T, D_u]:
        return types.numpy.simple.control_input_sequence(array)

    @staticmethod
    def control_input_batch[T: int, D_u: int, M: int](
        array: Array[Dims[T, D_u, M]],
    ) -> NumPyControlInputBatch[T, D_u, M]:
        return types.numpy.simple.control_input_batch(array)

    @staticmethod
    def obstacle_ids[K: int](
        array: NumberArray[Dims[K]] | Sequence[int],
    ) -> NumPyObstacleIds[K]:
        return types.numpy.obstacle_ids.create(ids=np.asarray(array))

    @staticmethod
    def simple_obstacle_states[T: int, D_o: int, K: int](
        *,
        states: Array[Dims[T, D_o, K]],
        covariance: Array[Dims[T, D_o, D_o, K]] | None = None,
    ) -> NumPySimpleObstacleStates[T, D_o, K]:
        return types.numpy.simple.obstacle_states.create(
            states=states, covariance=covariance
        )

    @staticmethod
    def obstacle_2d_poses[T: int, K: int](
        *,
        x: Array[Dims[T, K]],
        y: Array[Dims[T, K]],
        heading: Array[Dims[T, K]] | None = None,
        covariance: Array[Dims[T, D_o, D_o, K]] | None = None,
    ) -> NumPyObstacle2dPoses[T, K]:
        return types.numpy.obstacle_2d_poses.create(
            x=x,
            y=y,
            heading=heading if heading is not None else np.zeros_like(x),
            covariance=covariance,
        )

    @staticmethod
    def obstacle_2d_poses_for_time_step[K: int](
        *,
        x: Array[Dims[K]],
        y: Array[Dims[K]],
        heading: Array[Dims[K]] | None = None,
    ) -> NumPyObstacle2dPosesForTimeStep[K]:
        return types.numpy.obstacle_2d_poses_for_time_step.create(
            x=x,
            y=y,
            heading=heading if heading is not None else np.zeros_like(x),
        )

    @staticmethod
    def obstacle_2d_pose_samples[T: int, K: int, N: int](
        *,
        x: Array[Dims[T, K, N]],
        y: Array[Dims[T, K, N]],
        heading: Array[Dims[T, K, N]] | None = None,
    ) -> NumPySampledObstacle2dPoses[T, K, N]:
        return types.numpy.obstacle_2d_poses.sampled(
            x=x,
            y=y,
            heading=heading if heading is not None else np.zeros_like(x),
        )

    @staticmethod
    def distance[T: int, V: int, M: int, N: int](
        array: Array[Dims[T, V, M, N]],
    ) -> NumPyDistance[T, V, M, N]:
        return types.numpy.distance(array)

    @staticmethod
    def boundary_distance[T: int, M: int](
        array: Array[Dims[T, M]],
    ) -> NumPyBoundaryDistance[T, M]:
        return types.numpy.boundary_distance(array)


class jax:
    @staticmethod
    def state[D_x: int](array: Array[Dims[D_x]]) -> JaxState[D_x]:
        return types.jax.simple.state(jnp.asarray(array))

    @staticmethod
    def obstacle_ids[K: int](
        array: NumberArray[Dims[K]] | Sequence[int],
    ) -> JaxObstacleIds[K]:
        return types.jax.obstacle_ids.create(ids=jnp.asarray(array))

    @staticmethod
    def state_batch[T: int, D_x: int, M: int](
        array: Array[Dims[T, D_x, M]],
    ) -> JaxStateBatch[T, D_x, M]:
        return types.jax.simple.state_batch(jnp.asarray(array))

    @staticmethod
    def control_input_sequence[T: int, D_u: int](
        array: Array[Dims[T, D_u]],
    ) -> JaxControlInputSequence[T, D_u]:
        return types.jax.simple.control_input_sequence(jnp.asarray(array))

    @staticmethod
    def control_input_batch[T: int, D_u: int, M: int](
        array: Array[Dims[T, D_u, M]] | Float[JaxArray, "T D_u M"],
    ) -> JaxControlInputBatch[T, D_u, M]:
        return types.jax.simple.control_input_batch.create(array=jnp.asarray(array))

    @staticmethod
    def simple_obstacle_states[T: int, D_o: int, K: int](
        *,
        states: Array[Dims[T, D_o, K]] | Float[JaxArray, "T D_o K"],
        covariance: Array[Dims[T, D_o, D_o, K]]
        | Float[JaxArray, "T D_o D_o K"]
        | None = None,
    ) -> JaxSimpleObstacleStates[T, D_o, K]:
        return types.jax.simple.obstacle_states.create(
            states=jnp.asarray(states),
            covariance=jnp.asarray(covariance) if covariance is not None else None,
        )

    @staticmethod
    def obstacle_2d_poses[T: int, K: int](
        *,
        x: Array[Dims[T, K]] | Float[JaxArray, "T K"],
        y: Array[Dims[T, K]] | Float[JaxArray, "T K"],
        heading: Array[Dims[T, K]] | Float[JaxArray, "T K"] | None = None,
        covariance: Array[Dims[T, D_o, D_o, K]]
        | Float[JaxArray, f"T {D_O} {D_O} K"]
        | None = None,
    ) -> JaxObstacle2dPoses[T, K]:
        return types.jax.obstacle_2d_poses.create(
            x=jnp.asarray(x),
            y=jnp.asarray(y),
            heading=jnp.asarray(heading) if heading is not None else jnp.zeros_like(x),
            covariance=jnp.asarray(covariance) if covariance is not None else None,
        )

    @staticmethod
    def obstacle_2d_poses_for_time_step[K: int](
        *,
        x: Array[Dims[K]] | Float[JaxArray, "K"],
        y: Array[Dims[K]] | Float[JaxArray, "K"],
        heading: Array[Dims[K]] | Float[JaxArray, "K"] | None = None,
    ) -> JaxObstacle2dPosesForTimeStep[K]:
        return types.jax.obstacle_2d_poses_for_time_step.create(
            x=jnp.asarray(x),
            y=jnp.asarray(y),
            heading=jnp.asarray(heading) if heading is not None else jnp.zeros_like(x),
        )

    @staticmethod
    def obstacle_2d_pose_samples[T: int, K: int, N: int](
        *,
        x: Array[Dims[T, K, N]] | Float[JaxArray, "T K N"],
        y: Array[Dims[T, K, N]] | Float[JaxArray, "T K N"],
        heading: Array[Dims[T, K, N]] | Float[JaxArray, "T K N"] | None = None,
    ) -> JaxSampledObstacle2dPoses[T, K, N]:
        return types.jax.obstacle_2d_poses.sampled(
            x=jnp.asarray(x),
            y=jnp.asarray(y),
            heading=jnp.asarray(heading) if heading is not None else jnp.zeros_like(x),
        )

    @staticmethod
    def distance[T: int, V: int, M: int, N: int](
        array: Array[Dims[T, V, M, N]] | Float[JaxArray, "T V M N"],
    ) -> JaxDistance[T, V, M, N]:
        return types.jax.distance(jnp.asarray(array))

    @staticmethod
    def boundary_distance[T: int, M: int](
        array: Array[Dims[T, M]] | Float[JaxArray, "T M"],
    ) -> JaxBoundaryDistance[T, M]:
        return types.jax.boundary_distance(jnp.asarray(array))
