from typing import Protocol, Final

from trajax.type import DataType

from numtypes import Array, Dims, D

D_X: Final = 4
D_U: Final = 2

type D_x = D[4]
type D_u = D[2]


class State(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[D_x]]:
        """Returns the state as a NumPy array."""
        ...

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


class StateSequence[T: int = int](Protocol):
    def step(self, index: int) -> State:
        """Returns the state at the given time step index."""
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

    def rollout(self, index: int) -> StateSequence[T]:
        """Returns a single rollout from the batch as a state sequence."""
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


class ControlInputSequence[T: int = int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u]]:
        """Returns the control input sequence as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the control input sequence."""
        ...

    @property
    def dimension(self) -> D_u:
        """Control input dimension."""
        ...


class ControlInputBatch[T: int = int, M: int = int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, M]]:
        """Returns the control inputs as a NumPy array."""
        ...

    @property
    def rollout_count(self) -> M:
        """Number of rollouts in the batch."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the control inputs."""
        ...


class KinematicBicycleModel[
    InStateT: State,
    OutStateT: State,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
](Protocol):
    async def simulate(
        self, inputs: ControlInputBatchT, initial_state: InStateT
    ) -> StateBatchT:
        """Simulates the kinematic bicycle model over the given control inputs starting from the
        provided initial state."""
        ...

    async def step(self, input: ControlInputSequenceT, state: InStateT) -> OutStateT:
        """Simulates a single time step of the kinematic bicycle model given the control input
        and current state."""
        ...
