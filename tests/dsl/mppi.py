from trajax import types

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dims

import jax.numpy as jnp


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
type NumPyDistance[T: int, V: int, M: int] = types.numpy.Distance[T, V, M]

type JaxState[D_x: int] = types.jax.simple.State[D_x]
type JaxStateBatch[T: int, D_x: int, M: int] = types.jax.simple.StateBatch[T, D_x, M]
type JaxControlInputSequence[T: int, D_u: int] = types.jax.simple.ControlInputSequence[
    T, D_u
]
type JaxControlInputBatch[T: int, D_u: int, M: int] = (
    types.jax.simple.ControlInputBatch[T, D_u, M]
)
type JaxDistance[T: int, V: int, M: int] = types.jax.Distance[T, V, M]


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
    def distance[T: int, V: int, M: int](
        array: Array[Dims[T, V, M]],
    ) -> NumPyDistance[T, V, M]:
        return types.numpy.distance(array)


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
        return types.jax.simple.control_input_batch.create(array=jnp.asarray(array))

    @staticmethod
    def distance[T: int, V: int, M: int](
        array: Array[Dims[T, V, M]],
    ) -> JaxDistance[T, V, M]:
        return types.jax.distance(jnp.asarray(array))
