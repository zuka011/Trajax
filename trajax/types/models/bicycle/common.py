from typing import Protocol, Final

from trajax.types.array import DataType

from numtypes import Array, Dims, D

BICYCLE_D_X: Final = 4
BICYCLE_D_V: Final = 2
BICYCLE_D_U: Final = 2
BICYCLE_D_O: Final = 3

type BicycleD_x = D[4]
type BicycleD_v = D[2]
type BicycleD_u = D[2]
type BicycleD_o = D[3]


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
    def heading(self) -> float:
        """Orientation of the agent."""
        ...

    @property
    def speed(self) -> float:
        """Velocity of the agent."""
        ...

    @property
    def dimension(self) -> BicycleD_x:
        """State dimension."""
        ...


class BicycleStateSequence(Protocol):
    def step(self, index: int) -> BicycleState:
        """Returns the state at the given time step index."""
        ...


class BicycleStateBatch[T: int, M: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, BicycleD_x, M]]:
        """Returns the states as a NumPy array."""
        ...

    def heading(self) -> Array[Dims[T, M]]:
        """Returns the headings (orientations) of the states in the batch."""
        ...

    def speed(self) -> Array[Dims[T, M]]:
        """Returns the speeds of the states in the batch."""
        ...

    def rollout(self, index: int) -> BicycleStateSequence:
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
