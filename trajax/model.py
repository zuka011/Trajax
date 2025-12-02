from typing import Protocol, Final

from numtypes import Array, Dims, D
import numpy as np

D_X: Final = 4
D_U: Final = 2

type D_x = D[4]
type D_u = D[2]


class State(Protocol):
    @property
    def x(self) -> float:
        """X position of the agent."""
        ...

    @property
    def y(self) -> float:
        """Y position of the agent."""
        ...

    @property
    def theta(self) -> float:
        """Orientation of the agent."""
        ...

    @property
    def v(self) -> float:
        """Velocity of the agent."""
        ...


class StateBatch[T: int, M: int](Protocol):
    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, D_x, M]]:
        """Returns the states as a NumPy array."""
        ...

    def orientations(self) -> Array[Dims[T, M]]:
        """Returns the orientations of the states in the batch."""
        ...

    def velocities(self) -> Array[Dims[T, M]]:
        """Returns the velocities of the states in the batch."""
        ...

    @property
    def positions(self) -> "Positions[T, M] ":
        """Returns the positions of the states in the batch."""
        ...


class Positions[T: int, M: int](Protocol):
    def __array__(self, dtype: np.dtype | None = None) -> Array[Dims[T, D_u, M]]:
        """Returns the positions as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, M]]:
        """Returns the x positions."""
        ...

    def y(self) -> Array[Dims[T, M]]:
        """Returns the y positions."""
        ...


class ControlInputBatch[T: int, M: int](Protocol):
    @property
    def rollout_count(self) -> M:
        """Number of rollouts in the batch."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the control inputs."""
        ...


class DynamicalModel(Protocol):
    async def simulate[T: int, M: int](
        self,
        inputs: ControlInputBatch[T, M],
        initial_state: State,
    ) -> StateBatch[T, M]:
        """Simulates the dynamical model over the given control inputs starting from the
        provided initial state."""
        ...
