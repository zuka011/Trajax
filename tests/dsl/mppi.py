from trajax import types

from jaxtyping import Array as JaxArray, Float
from numtypes import Array, Dims
import jax.numpy as jnp


class numpy:
    @staticmethod
    def state[D_x: int](array: Array[Dims[D_x]]) -> types.numpy.basic.State[D_x]:
        return types.numpy.basic.state(array)

    @staticmethod
    def state_batch[T: int, D_x: int, M: int](
        array: Array[Dims[T, D_x, M]],
    ) -> types.numpy.basic.StateBatch[T, D_x, M]:
        return types.numpy.basic.state_batch(array)

    @staticmethod
    def control_input_sequence[T: int, D_u: int](
        array: Array[Dims[T, D_u]],
    ) -> types.numpy.basic.ControlInputSequence[T, D_u]:
        return types.numpy.basic.control_input_sequence(array)

    @staticmethod
    def control_input_batch[T: int, D_u: int, M: int](
        array: Array[Dims[T, D_u, M]],
    ) -> types.numpy.basic.ControlInputBatch[T, D_u, M]:
        return types.numpy.basic.control_input_batch(array)


class jax:
    @staticmethod
    def state[D_x: int](array: Array[Dims[D_x]]) -> types.jax.basic.State[D_x]:
        return types.jax.basic.state(jnp.asarray(array))

    @staticmethod
    def state_batch[T: int, D_x: int, M: int](
        array: Array[Dims[T, D_x, M]],
    ) -> types.jax.basic.StateBatch[T, D_x, M]:
        return types.jax.basic.state_batch(jnp.asarray(array))

    @staticmethod
    def control_input_sequence[T: int, D_u: int](
        array: Array[Dims[T, D_u]],
    ) -> types.jax.basic.ControlInputSequence[T, D_u]:
        return types.jax.basic.control_input_sequence(jnp.asarray(array))

    @staticmethod
    def control_input_batch[T: int, D_u: int, M: int](
        array: Array[Dims[T, D_u, M]] | Float[JaxArray, "T D_u M"],
    ) -> types.jax.basic.ControlInputBatch[T, D_u, M]:
        return types.jax.basic.control_input_batch(jnp.asarray(array))
