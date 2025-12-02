from typing import cast
from dataclasses import dataclass

from trajax.model import D_X, D_x, D_U, D_u, State, ControlInputBatch
from trajax.type import jaxtyped

from jaxtyping import Array, Float
from numtypes import Array as NumpyArray, Dims

import jax
import jax.numpy as jnp
import numpy as np


type StateArray = Float[Array, f"{D_X}"]
type ControlInputArray = Float[Array, f"{D_U}"]

type StateBatchArray = Float[Array, f"T {D_X} M"]
type ControlInputBatchArray = Float[Array, f"T {D_U} M"]


@jaxtyped
@dataclass(frozen=True)
class JaxState:
    state: StateArray

    @property
    def x(self) -> float:
        return float(self.state[0])

    @property
    def y(self) -> float:
        return float(self.state[1])

    @property
    def theta(self) -> float:
        return float(self.state[2])

    @property
    def v(self) -> float:
        return float(self.state[3])


@jaxtyped
@dataclass(frozen=True)
class JaxStateBatch[T: int, M: int]:
    states: StateBatchArray

    def __array__(self, dtype: np.dtype | None = None) -> NumpyArray[Dims[T, D_x, M]]:
        return np.asarray(self.states, dtype=dtype)

    def orientations(self) -> NumpyArray[Dims[T, M]]:
        return np.asarray(self.states[:, 2, :])

    def velocities(self) -> NumpyArray[Dims[T, M]]:
        return np.asarray(self.states[:, 3, :])

    @property
    def positions(self) -> "JaxPositions[T, M]":
        return JaxPositions(state=self)


@jaxtyped
@dataclass(frozen=True)
class JaxPositions[T: int, M: int]:
    state: JaxStateBatch

    def __array__(self, dtype: np.dtype | None = None) -> NumpyArray[Dims[T, D_u, M]]:
        return np.asarray(self.state.states[:, :2, :], dtype=dtype)

    def x(self) -> NumpyArray[Dims[T, M]]:
        return np.asarray(self.state.states[:, 0, :])

    def y(self) -> NumpyArray[Dims[T, M]]:
        return np.asarray(self.state.states[:, 1, :])


@jaxtyped
@dataclass(frozen=True)
class JaxControlInputBatch[T: int, M: int]:
    inputs: ControlInputBatchArray

    @property
    def rollout_count(self) -> M:
        return cast(M, self.inputs.shape[2])

    @property
    def horizon(self) -> T:
        return cast(T, self.inputs.shape[0])


@dataclass(frozen=True)
class JaxBicycleModel:
    time_step_size: float

    async def simulate[T: int, M: int](
        self,
        inputs: ControlInputBatch[T, M],
        initial_state: State,
    ) -> JaxStateBatch[T, M]:
        assert isinstance(inputs, JaxControlInputBatch), (
            "Only JAX inputs are supported."
        )
        assert isinstance(initial_state, JaxState), (
            "Only JAX initial states are supported."
        )

        rollout_count = inputs.rollout_count

        initial = jnp.stack(
            [
                jnp.full(rollout_count, initial_state.x),
                jnp.full(rollout_count, initial_state.y),
                jnp.full(rollout_count, initial_state.theta),
                jnp.full(rollout_count, initial_state.v),
            ]
        )

        return JaxStateBatch(
            simulate_jit(inputs.inputs, initial, time_step_size=self.time_step_size)
        )


@jax.jit
def simulate_jit(
    inputs: ControlInputBatchArray,
    initial: Float[Array, "4 M"],
    *,
    time_step_size: float,
) -> StateBatchArray:
    def step(
        state: Float[Array, "4 M"], control: Float[Array, "2 M"]
    ) -> tuple[Float[Array, "4 M"], Float[Array, "4 M"]]:
        x, y, theta, v = state[0], state[1], state[2], state[3]
        acceleration, steering = control[0], control[1]

        new_x = x + v * jnp.cos(theta) * time_step_size
        new_y = y + v * jnp.sin(theta) * time_step_size
        new_theta = theta + steering * time_step_size
        new_v = v + acceleration * time_step_size

        new_state = jnp.stack([new_x, new_y, new_theta, new_v])
        return new_state, new_state

    _, states = jax.lax.scan(step, initial, inputs)
    return states
