from typing import overload

from trajax import (
    D_X,
    D_U,
    NumpyState,
    NumpyControlInputBatch,
    JaxState,
    JaxControlInputBatch,
)

from numtypes import array, Array, Dims, shape_of
import jax.numpy as jnp
import numpy as np


class numpy:
    @overload
    @staticmethod
    def control_inputs(
        *,
        rollout_count: int,
        time_horizon: int,
        acceleration: float,
        steering: float,
    ) -> NumpyControlInputBatch: ...

    @overload
    @staticmethod
    def control_inputs[T: int](
        *,
        rollout_count: int,
        acceleration: Array[Dims[T]],
        steering: Array[Dims[T]],
    ) -> NumpyControlInputBatch: ...

    @staticmethod
    def control_inputs(
        *,
        rollout_count: int,
        time_horizon: int | None = None,
        acceleration: float | Array,
        steering: float | Array,
    ) -> NumpyControlInputBatch:
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
                    inputs, matches=(T, D_U, rollout_count), name="control inputs"
                )

                return NumpyControlInputBatch(inputs)
            case float() | int(), float() | int():
                assert time_horizon is not None, (
                    "time_horizon must be provided when passing constant inputs."
                )

                return NumpyControlInputBatch(
                    array(
                        [
                            [
                                [acceleration] * rollout_count,
                                [steering] * rollout_count,
                            ],
                        ]
                        * time_horizon,
                        shape=(time_horizon, D_U, rollout_count),
                    )
                )
            case _:
                assert False, (
                    f"Received invalid combination of arguments. "
                    f"Acceleration: {acceleration}, Steering: {steering}"
                )

    @staticmethod
    def state(*, x: float, y: float, theta: float, v: float) -> NumpyState:
        return NumpyState(array([x, y, theta, v], shape=(D_X,)))


class jax:
    @overload
    @staticmethod
    def control_inputs(
        *,
        rollout_count: int,
        time_horizon: int,
        acceleration: float,
        steering: float,
    ) -> JaxControlInputBatch: ...

    @overload
    @staticmethod
    def control_inputs[T: int](
        *,
        rollout_count: int,
        acceleration: Array[Dims[T]],
        steering: Array[Dims[T]],
    ) -> JaxControlInputBatch: ...

    @staticmethod
    def control_inputs(
        *,
        rollout_count: int,
        time_horizon: int | None = None,
        acceleration: float | Array,
        steering: float | Array,
    ) -> JaxControlInputBatch:
        return JaxControlInputBatch(
            jnp.array(
                numpy.control_inputs(
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
