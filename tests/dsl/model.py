from typing import overload

from trajax import (
    bicycle,
    NumPyState,
    NumPyControlInputBatch,
    JaxState,
    JaxControlInputBatch,
)

from numtypes import array, Array, Dims, shape_of
import jax.numpy as jnp
import numpy as np


class numpy:
    @overload
    @staticmethod
    def control_input_batch[T: int, M: int](
        *,
        time_horizon: T,
        rollout_count: M,
        acceleration: float,
        steering: float,
    ) -> NumPyControlInputBatch[T, M]: ...

    @overload
    @staticmethod
    def control_input_batch[T: int, M: int](
        *,
        rollout_count: M,
        acceleration: Array[Dims[T]],
        steering: Array[Dims[T]],
    ) -> NumPyControlInputBatch[T, M]: ...

    @staticmethod
    def control_input_batch(
        *,
        rollout_count: int,
        time_horizon: int | None = None,
        acceleration: float | Array,
        steering: float | Array,
    ) -> NumPyControlInputBatch:
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
                    matches=(T, bicycle.D_U, rollout_count),
                    name="control inputs",
                )

                return NumPyControlInputBatch(inputs)
            case float() | int(), float() | int():
                assert time_horizon is not None, (
                    "time_horizon must be provided when passing constant inputs."
                )

                return NumPyControlInputBatch(
                    array(
                        [
                            [
                                [acceleration] * rollout_count,
                                [steering] * rollout_count,
                            ],
                        ]
                        * time_horizon,
                        shape=(time_horizon, bicycle.D_U, rollout_count),
                    )
                )
            case _:
                assert False, (
                    f"Received invalid combination of arguments. "
                    f"Acceleration: {acceleration}, Steering: {steering}"
                )

    @staticmethod
    def state(*, x: float, y: float, theta: float, v: float) -> NumPyState:
        return NumPyState(array([x, y, theta, v], shape=(bicycle.D_X,)))


class jax:
    @overload
    @staticmethod
    def control_input_batch[T: int, M: int](
        *,
        time_horizon: T,
        rollout_count: M,
        acceleration: float,
        steering: float,
    ) -> JaxControlInputBatch[T, M]: ...

    @overload
    @staticmethod
    def control_input_batch[T: int, M: int](
        *,
        rollout_count: M,
        acceleration: Array[Dims[T]],
        steering: Array[Dims[T]],
    ) -> JaxControlInputBatch[T, M]: ...

    @staticmethod
    def control_input_batch(
        *,
        rollout_count: int,
        time_horizon: int | None = None,
        acceleration: float | Array,
        steering: float | Array,
    ) -> JaxControlInputBatch:
        return JaxControlInputBatch(
            jnp.array(
                numpy.control_input_batch(
                    rollout_count=rollout_count,
                    time_horizon=time_horizon,  # type: ignore
                    acceleration=acceleration,  # type: ignore
                    steering=steering,  # type: ignore
                ).inputs
            )
        )

    @staticmethod
    def state(*, x: float, y: float, theta: float, v: float) -> JaxState:
        return JaxState(jnp.array([x, y, theta, v]))
