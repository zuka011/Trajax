from dataclasses import dataclass

from trajax.model import D_x, D_X, D_u, State, ControlInputBatch

from numtypes import Array, Dims, D, shape_of
import numpy as np


type StateArray = Array[Dims[D_x]]
type ControlInputArray = Array[Dims[D_u]]

type StateBatchArray[T: int, M: int] = Array[Dims[T, D_x, M]]
type ControlInputBatchArray[T: int, M: int] = Array[Dims[T, D_u, M]]


@dataclass(frozen=True)
class NumpyState:
    state: StateArray

    @property
    def x(self) -> float:
        return self.state[0]

    @property
    def y(self) -> float:
        return self.state[1]

    @property
    def theta(self) -> float:
        return self.state[2]

    @property
    def v(self) -> float:
        return self.state[3]


@dataclass(frozen=True)
class NumpyStateBatch[T: int, M: int]:
    states: StateBatchArray[T, M]

    def __array__(self, dtype: np.dtype | None = None) -> StateBatchArray[T, M]:
        return self.states

    def orientations(self) -> Array[Dims[T, M]]:
        return self.states[:, 2, :]

    def velocities(self) -> Array[Dims[T, M]]:
        return self.states[:, 3, :]

    @property
    def positions(self) -> "NumpyPositions":
        return NumpyPositions(state=self)


@dataclass(frozen=True)
class NumpyPositions[T: int, M: int]:
    state: NumpyStateBatch[T, M]

    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, D[2], M]]:
        return self.state.states[:, :2, :]

    def x(self) -> Array[Dims[T, M]]:
        return self.state.states[:, 0, :]

    def y(self) -> Array[Dims[T, M]]:
        return self.state.states[:, 1, :]


@dataclass(frozen=True)
class NumpyControlInputBatch[T: int, M: int]:
    inputs: ControlInputBatchArray[T, M]

    @property
    def rollout_count(self) -> M:
        return self.inputs.shape[2]

    @property
    def horizon(self) -> T:
        return self.inputs.shape[0]


@dataclass(frozen=True)
class BicycleModel:
    time_step_size: float

    async def simulate[T: int, M: int](
        self,
        inputs: ControlInputBatch[T, M],
        initial_state: State,
    ) -> NumpyStateBatch[T, M]:
        assert isinstance(inputs, NumpyControlInputBatch), (
            "Only NumPy inputs are supported."
        )
        assert isinstance(initial_state, NumpyState), (
            "Only NumPy initial states are supported."
        )

        horizon = inputs.horizon
        rollout_count = inputs.rollout_count
        states = np.zeros((horizon, 4, rollout_count))

        x = np.full(rollout_count, initial_state.x)
        y = np.full(rollout_count, initial_state.y)
        theta = np.full(rollout_count, initial_state.theta)
        v = np.full(rollout_count, initial_state.v)

        for t in range(horizon):
            acceleration = inputs.inputs[t, 0, :]
            steering = inputs.inputs[t, 1, :]

            x = x + v * np.cos(theta) * self.time_step_size
            y = y + v * np.sin(theta) * self.time_step_size
            theta = theta + steering * self.time_step_size
            v = v + acceleration * self.time_step_size

            states[t, 0, :] = x
            states[t, 1, :] = y
            states[t, 2, :] = theta
            states[t, 3, :] = v

        assert shape_of(
            states, matches=(horizon, D_X, rollout_count), name="simulated states"
        )

        return NumpyStateBatch(states)
