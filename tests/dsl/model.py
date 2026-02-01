from typing import overload

from trajax import types

from numtypes import array, Array, Dims, shape_of

import jax.numpy as jnp
import numpy as np


type NumPyBicycleState = types.numpy.bicycle.State
type NumPyBicycleControlInputBatch[T: int, M: int] = (
    types.numpy.bicycle.ControlInputBatch[T, M]
)
type NumPyUnicycleState = types.numpy.unicycle.State
type NumPyUnicycleControlInputBatch[T: int, M: int] = (
    types.numpy.unicycle.ControlInputBatch[T, M]
)

type JaxBicycleState = types.jax.bicycle.State
type JaxBicycleControlInputBatch[T: int, M: int] = types.jax.bicycle.ControlInputBatch[
    T, M
]
type JaxUnicycleState = types.jax.unicycle.State
type JaxUnicycleControlInputBatch[T: int, M: int] = (
    types.jax.unicycle.ControlInputBatch[T, M]
)


class numpy:
    class bicycle:
        @overload
        @staticmethod
        def control_input_batch[T: int, M: int](
            *,
            time_horizon: T,
            rollout_count: M,
            acceleration: float,
            steering: float,
        ) -> NumPyBicycleControlInputBatch[T, M]: ...

        @overload
        @staticmethod
        def control_input_batch[T: int, M: int](
            *,
            rollout_count: M,
            acceleration: Array[Dims[T]],
            steering: Array[Dims[T]],
        ) -> NumPyBicycleControlInputBatch[T, M]: ...

        @staticmethod
        def control_input_batch(
            *,
            rollout_count: int,
            time_horizon: int | None = None,
            acceleration: float | Array,
            steering: float | Array,
        ) -> NumPyBicycleControlInputBatch:
            match acceleration, steering:
                case np.ndarray(), np.ndarray():
                    assert time_horizon is None, (
                        "time_horizon should not be provided when passing sequences."
                    )
                    assert acceleration.shape == steering.shape, (
                        f"Acceleration and steering sequences must have the same shape. Got "
                        f"{acceleration.shape} (acceleration) and {steering.shape} (steering)."
                    )

                    T = acceleration.shape[0]
                    inputs = np.array(
                        [[acceleration.tolist(), steering.tolist()]] * rollout_count
                    ).transpose(2, 1, 0)

                    assert shape_of(
                        inputs,
                        matches=(T, types.bicycle.D_U, rollout_count),
                        name="control inputs",
                    )

                    return types.numpy.bicycle.control_input_batch(inputs)
                case float() | int(), float() | int():
                    assert time_horizon is not None, (
                        "time_horizon must be provided when passing constant inputs."
                    )

                    return types.numpy.bicycle.control_input_batch(
                        array(
                            [
                                [
                                    [acceleration] * rollout_count,
                                    [steering] * rollout_count,
                                ],
                            ]
                            * time_horizon,
                            shape=(time_horizon, types.bicycle.D_U, rollout_count),
                        )
                    )
                case _:
                    assert False, (
                        f"Received invalid combination of arguments. "
                        f"Acceleration: {acceleration}, Steering: {steering}"
                    )

        @staticmethod
        def state(
            *, x: float, y: float, heading: float, speed: float
        ) -> NumPyBicycleState:
            return types.numpy.bicycle.state.create(
                x=x, y=y, heading=heading, speed=speed
            )

    class unicycle:
        @overload
        @staticmethod
        def control_input_batch[T: int, M: int](
            *,
            time_horizon: T,
            rollout_count: M,
            linear_velocity: float,
            angular_velocity: float,
        ) -> NumPyUnicycleControlInputBatch[T, M]: ...

        @overload
        @staticmethod
        def control_input_batch[T: int, M: int](
            *,
            rollout_count: M,
            linear_velocity: Array[Dims[T]],
            angular_velocity: Array[Dims[T]],
        ) -> NumPyUnicycleControlInputBatch[T, M]: ...

        @staticmethod
        def control_input_batch(
            *,
            rollout_count: int,
            time_horizon: int | None = None,
            linear_velocity: float | Array,
            angular_velocity: float | Array,
        ) -> NumPyUnicycleControlInputBatch:
            match linear_velocity, angular_velocity:
                case np.ndarray(), np.ndarray():
                    assert time_horizon is None, (
                        "time_horizon should not be provided when passing sequences."
                    )
                    assert linear_velocity.shape == angular_velocity.shape, (
                        f"Linear velocity and angular velocity sequences must have the same shape. Got "
                        f"{linear_velocity.shape} (linear_velocity) and {angular_velocity.shape} (angular_velocity)."
                    )

                    T = linear_velocity.shape[0]
                    inputs = np.array(
                        [[linear_velocity.tolist(), angular_velocity.tolist()]]
                        * rollout_count
                    ).transpose(2, 1, 0)

                    assert shape_of(
                        inputs,
                        matches=(T, types.unicycle.D_U, rollout_count),
                        name="control inputs",
                    )

                    return types.numpy.unicycle.control_input_batch(inputs)
                case float() | int(), float() | int():
                    assert time_horizon is not None, (
                        "time_horizon must be provided when passing constant inputs."
                    )

                    return types.numpy.unicycle.control_input_batch(
                        array(
                            [
                                [
                                    [linear_velocity] * rollout_count,
                                    [angular_velocity] * rollout_count,
                                ],
                            ]
                            * time_horizon,
                            shape=(time_horizon, types.unicycle.D_U, rollout_count),
                        )
                    )
                case _:
                    assert False, (
                        f"Received invalid combination of arguments. "
                        f"Linear velocity: {linear_velocity}, Angular velocity: {angular_velocity}"
                    )

        @staticmethod
        def state(*, x: float, y: float, heading: float) -> NumPyUnicycleState:
            return types.numpy.unicycle.state.create(x=x, y=y, heading=heading)


class jax:
    class bicycle:
        @overload
        @staticmethod
        def control_input_batch[T: int, M: int](
            *,
            time_horizon: T,
            rollout_count: M,
            acceleration: float,
            steering: float,
        ) -> JaxBicycleControlInputBatch[T, M]: ...

        @overload
        @staticmethod
        def control_input_batch[T: int, M: int](
            *,
            rollout_count: M,
            acceleration: Array[Dims[T]],
            steering: Array[Dims[T]],
        ) -> JaxBicycleControlInputBatch[T, M]: ...

        @staticmethod
        def control_input_batch(
            *,
            rollout_count: int,
            time_horizon: int | None = None,
            acceleration: float | Array,
            steering: float | Array,
        ) -> JaxBicycleControlInputBatch:
            return types.jax.bicycle.control_input_batch.create(
                array=jnp.array(
                    numpy.bicycle.control_input_batch(
                        rollout_count=rollout_count,
                        time_horizon=time_horizon,  # type: ignore
                        acceleration=acceleration,  # type: ignore
                        steering=steering,  # type: ignore
                    )
                )
            )

        @staticmethod
        def state(
            *, x: float, y: float, heading: float, speed: float
        ) -> JaxBicycleState:
            return types.jax.bicycle.state.create(
                x=x, y=y, heading=heading, speed=speed
            )

    class unicycle:
        @overload
        @staticmethod
        def control_input_batch[T: int, M: int](
            *,
            time_horizon: T,
            rollout_count: M,
            linear_velocity: float,
            angular_velocity: float,
        ) -> JaxUnicycleControlInputBatch[T, M]: ...

        @overload
        @staticmethod
        def control_input_batch[T: int, M: int](
            *,
            rollout_count: M,
            linear_velocity: Array[Dims[T]],
            angular_velocity: Array[Dims[T]],
        ) -> JaxUnicycleControlInputBatch[T, M]: ...

        @staticmethod
        def control_input_batch(
            *,
            rollout_count: int,
            time_horizon: int | None = None,
            linear_velocity: float | Array,
            angular_velocity: float | Array,
        ) -> JaxUnicycleControlInputBatch:
            return types.jax.unicycle.control_input_batch.create(
                array=jnp.array(
                    numpy.unicycle.control_input_batch(
                        rollout_count=rollout_count,
                        time_horizon=time_horizon,  # type: ignore
                        linear_velocity=linear_velocity,  # type: ignore
                        angular_velocity=angular_velocity,  # type: ignore
                    )
                )
            )

        @staticmethod
        def state(*, x: float, y: float, heading: float) -> JaxUnicycleState:
            return types.jax.unicycle.state.create(x=x, y=y, heading=heading)
