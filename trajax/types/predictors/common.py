from typing import Protocol
from dataclasses import dataclass

from trajax.types.array import DataType
from trajax.types.costs import D_o

from numtypes import Array, Dims


@dataclass(kw_only=True, frozen=True)
class EstimatedObstacleStates[StatesT, VelocitiesT]:
    states: StatesT
    velocities: VelocitiesT


class ObstacleStatesHistory[T: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        """Returns the obstacle history as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon of the obstacle history."""
        ...

    @property
    def dimension(self) -> D_o:
        """The dimension of the obstacle states."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class ObstacleModel[HistoryT, StatesT, VelocitiesT, InputSequencesT, StateSequencesT](
    Protocol
):
    def estimate_state_from(
        self, history: HistoryT
    ) -> EstimatedObstacleStates[StatesT, VelocitiesT]:
        """Estimates the current states and velocities of objects given their history."""
        ...

    def input_to_maintain(
        self, velocities: VelocitiesT, *, states: StatesT, horizon: int
    ) -> InputSequencesT:
        """Generates control inputs to maintain the given velocities over the specified horizon."""
        ...

    def forward(self, *, current: StatesT, input: InputSequencesT) -> StateSequencesT:
        """Simulates the objects forward in time given the current states and control inputs."""
        ...


class EmptyPredictionCreator[PredictionT](Protocol):
    def __call__(self, *, horizon: int) -> PredictionT:
        """Creates an empty prediction with the specified horizon."""
        ...


class ObstacleMotionPredictor[HistoryT, PredictionT](Protocol):
    def predict(self, *, history: HistoryT) -> PredictionT:
        """Predicts the future states of obstacles based on their state history."""
        ...
