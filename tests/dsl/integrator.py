from typing import Final
from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp

from tests.dsl.mppi import (
    SimpleNumPyControlInputSequence,
    SimpleNumPyControlInputBatch,
    SimpleNumPyState,
    SimpleNumPyStateBatch,
    SimpleJaxControlInputSequence,
    SimpleJaxControlInputBatch,
    SimpleJaxState,
    SimpleJaxStateBatch,
)


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
        inputs: SimpleNumPyControlInputBatch[T, D_u, M],
        initial_state: SimpleNumPyState[D_x],
    ) -> SimpleNumPyStateBatch[T, D_x, M]:
        controls = np.asarray(inputs)
        initial = np.asarray(initial_state)

        states = initial[:, None] + np.cumsum(controls * self.time_step, axis=0)

        return SimpleNumPyStateBatch(array=states)

    async def step[T: int, D_u: int, D_x: int](
        self,
        input: SimpleNumPyControlInputSequence[T, D_u],
        state: SimpleNumPyState[D_x],
    ) -> SimpleNumPyState[D_x]:
        controls = np.asarray(input)
        current_state = np.asarray(state)

        new_state = current_state + controls[0] * self.time_step

        return SimpleNumPyState(array=new_state)


@dataclass(frozen=True)
class JaxIntegratorModel:
    time_step: float

    @staticmethod
    def create(*, time_step_size: float) -> "JaxIntegratorModel":
        return JaxIntegratorModel(time_step=time_step_size)

    async def simulate(
        self,
        inputs: SimpleJaxControlInputBatch,
        initial_state: SimpleJaxState,
    ) -> SimpleJaxStateBatch:
        controls = inputs.array
        initial = initial_state.array

        states = initial[:, None] + jnp.cumsum(controls * self.time_step, axis=0)

        return SimpleJaxStateBatch(array=states)

    async def step(
        self,
        input: SimpleJaxControlInputSequence,
        state: SimpleJaxState,
    ) -> SimpleJaxState:
        controls = input.array
        current_state = state.array

        new_state = current_state + controls[0] * self.time_step

        return SimpleJaxState(array=new_state)


class integrator:
    numpy: Final = NumPyIntegratorModel.create
    jax: Final = JaxIntegratorModel.create
