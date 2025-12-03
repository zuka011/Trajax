from typing import Protocol, Final

from trajax.type import DataType

from numtypes import Array, Dims, D

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


class StateBatch[T: int = int, M: int = int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_x, M]]:
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


class Positions[T: int = int, M: int = int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        """Returns the positions as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, M]]:
        """Returns the x positions."""
        ...

    def y(self) -> Array[Dims[T, M]]:
        """Returns the y positions."""
        ...


class ControlInputBatch[T: int = int, M: int = int](Protocol):
    @property
    def rollout_count(self) -> M:
        """Number of rollouts in the batch."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the control inputs."""
        ...


class KinematicBicycleModel[
    ControlInputBatchT: ControlInputBatch,
    StateT: State,
    StateBatchT: StateBatch,
](Protocol):
    async def simulate(
        self, inputs: ControlInputBatchT, initial_state: StateT
    ) -> StateBatchT:
        """Simulates the kinematic bicycle model over the given control inputs starting from the
        provided initial state."""
        ...
