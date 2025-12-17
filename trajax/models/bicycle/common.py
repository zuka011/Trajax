from typing import Protocol, Final

from trajax.type import DataType

from numtypes import Array, Dims, D

BICYCLE_D_X: Final = 4
BICYCLE_D_U: Final = 2

type BicycleD_x = D[4]
type BicycleD_u = D[2]


class BicycleState(Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[BicycleD_x]]:
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

    @property
    def dimension(self) -> BicycleD_x:
        """State dimension."""
        ...


class StateSequence(Protocol):
    def step(self, index: int) -> BicycleState:
        """Returns the state at the given time step index."""
        ...


class BicycleStateBatch[T: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_x, M]]:
        """Returns the states as a NumPy array."""
        ...

    def orientations(self) -> Array[Dims[T, M]]:
        """Returns the orientations of the states in the batch."""
        ...

    def velocities(self) -> Array[Dims[T, M]]:
        """Returns the velocities of the states in the batch."""
        ...

    def rollout(self, index: int) -> StateSequence:
        """Returns a single rollout from the batch as a state sequence."""
        ...

    @property
    def positions(self) -> "BicyclePositions[T, M] ":
        """Returns the positions of the states in the batch."""
        ...


class BicyclePositions[T: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_u, M]]:
        """Returns the positions as a NumPy array."""
        ...

    def x(self) -> Array[Dims[T, M]]:
        """Returns the x positions."""
        ...

    def y(self) -> Array[Dims[T, M]]:
        """Returns the y positions."""
        ...


class BicycleControlInputSequence[T: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_u]]:
        """Returns the control input sequence as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """Time horizon of the control input sequence."""
        ...

    @property
    def dimension(self) -> BicycleD_u:
        """Control input dimension."""
        ...


class BicycleControlInputBatch[T: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_u, M]]:
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


class BicycleModel[
    InStateT: BicycleState,
    OutStateT: BicycleState,
    StateBatchT: BicycleStateBatch,
    ControlInputSequenceT: BicycleControlInputSequence,
    ControlInputBatchT: BicycleControlInputBatch,
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
