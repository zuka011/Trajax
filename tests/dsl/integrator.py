from typing import Final
from dataclasses import dataclass

from trajax import types

import numpy as np
import jax.numpy as jnp


type NumpyState[D_x: int] = types.numpy.basic.State[D_x]
type NumpyStateBatch[T: int, D_x: int, M: int] = types.numpy.basic.StateBatch[T, D_x, M]
type NumpyControlInputSequence[T: int, D_u: int] = (
    types.numpy.basic.ControlInputSequence[T, D_u]
)
type NumpyControlInputBatch[T: int, D_u: int, M: int] = (
    types.numpy.basic.ControlInputBatch[T, D_u, M]
)

type JaxState[D_x: int] = types.jax.basic.State[D_x]
type JaxStateBatch[T: int, D_x: int, M: int] = types.jax.basic.StateBatch[T, D_x, M]
type JaxControlInputSequence[T: int, D_u: int] = types.jax.basic.ControlInputSequence[
    T, D_u
]
type JaxControlInputBatch[T: int, D_u: int, M: int] = types.jax.basic.ControlInputBatch[
    T, D_u, M
]


@dataclass(frozen=True)
class NumPyIntegratorModel:
    time_step: float

    @staticmethod
    def create(*, time_step_size: float) -> "NumPyIntegratorModel":
        """A simple integrator model where state = cumulative sum of controls.

        This model represents a particle that moves according to velocity commands.
        State dimension must equal control dimension (D_x == D_u).

        x_{t+1} = x_t + u_t * dt
        """
        return NumPyIntegratorModel(time_step=time_step_size)

    async def simulate[T: int, D_u: int, D_x: int, M: int](
        self,
        inputs: NumpyControlInputBatch[T, D_u, M],
        initial_state: NumpyState[D_x],
    ) -> NumpyStateBatch[T, D_x, M]:
        controls = np.asarray(inputs)
        initial = np.asarray(initial_state)

        states = initial[:, None] + np.cumsum(controls * self.time_step, axis=0)

        return types.numpy.basic.state_batch(array=states)

    async def step[T: int, D_u: int, D_x: int](
        self,
        input: NumpyControlInputSequence[T, D_u],
        state: NumpyState[D_x],
    ) -> NumpyState[D_x]:
        controls = np.asarray(input)
        current_state = np.asarray(state)

        new_state = current_state + controls[0] * self.time_step

        return types.numpy.basic.state(array=new_state)


@dataclass(frozen=True)
class JaxIntegratorModel:
    time_step: float

    @staticmethod
    def create(*, time_step_size: float) -> "JaxIntegratorModel":
        return JaxIntegratorModel(time_step=time_step_size)

    async def simulate(
        self,
        inputs: JaxControlInputBatch,
        initial_state: JaxState,
    ) -> JaxStateBatch:
        controls = inputs.array
        initial = initial_state.array

        states = initial[:, None] + jnp.cumsum(controls * self.time_step, axis=0)

        return types.jax.basic.state_batch(array=states)

    async def step(
        self,
        input: JaxControlInputSequence,
        state: JaxState,
    ) -> JaxState:
        controls = input.array
        current_state = state.array

        new_state = current_state + controls[0] * self.time_step

        return types.jax.basic.state(array=new_state)


class integrator:
    numpy: Final = NumPyIntegratorModel.create
    jax: Final = JaxIntegratorModel.create
