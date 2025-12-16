from trajax import types

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dims

import jax.numpy as jnp


type NumpyState[D_x: int] = types.numpy.simple.State[D_x]
type NumpyStateBatch[T: int, D_x: int, M: int] = types.numpy.simple.StateBatch[
    T, D_x, M
]
type NumpyControlInputSequence[T: int, D_u: int] = (
    types.numpy.simple.ControlInputSequence[T, D_u]
)
type NumpyControlInputBatch[T: int, D_u: int, M: int] = (
    types.numpy.simple.ControlInputBatch[T, D_u, M]
)

type JaxState[D_x: int] = types.jax.simple.State[D_x]
type JaxStateBatch[T: int, D_x: int, M: int] = types.jax.simple.StateBatch[T, D_x, M]
type JaxControlInputSequence[T: int, D_u: int] = types.jax.simple.ControlInputSequence[
    T, D_u
]
type JaxControlInputBatch[T: int, D_u: int, M: int] = (
    types.jax.simple.ControlInputBatch[T, D_u, M]
)


class numpy:
    @staticmethod
    def state[D_x: int](array: Array[Dims[D_x]]) -> NumpyState[D_x]:
        return types.numpy.simple.state(array)

    @staticmethod
    def state_batch[T: int, D_x: int, M: int](
        array: Array[Dims[T, D_x, M]],
    ) -> NumpyStateBatch[T, D_x, M]:
        return types.numpy.simple.state_batch(array)

    @staticmethod
    def control_input_sequence[T: int, D_u: int](
        array: Array[Dims[T, D_u]],
    ) -> NumpyControlInputSequence[T, D_u]:
        return types.numpy.simple.control_input_sequence(array)

    @staticmethod
    def control_input_batch[T: int, D_u: int, M: int](
        array: Array[Dims[T, D_u, M]],
    ) -> NumpyControlInputBatch[T, D_u, M]:
        return types.numpy.simple.control_input_batch(array)


class jax:
    @staticmethod
    def state[D_x: int](array: Array[Dims[D_x]]) -> JaxState[D_x]:
        return types.jax.simple.state(jnp.asarray(array))

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
        return types.jax.simple.control_input_batch(jnp.asarray(array))
