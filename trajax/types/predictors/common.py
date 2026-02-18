from typing import Protocol, Self, Any
from dataclasses import dataclass

from trajax.types.array import DataType

from numtypes import Array, IndexArray, Dims


@dataclass(kw_only=True, frozen=True)
class EstimatedObstacleStates[StatesT, InputsT, CovarianceT]:
    states: StatesT
    inputs: InputsT
    covariance: CovarianceT | None


class ObstacleStatesHistory[T: int, D_o: int, K: int, ObstacleStatesForTimeStepT = Any](
    Protocol
):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        """Returns the obstacle history as a NumPy array."""
        ...

    def last(self) -> ObstacleStatesForTimeStepT:
        """Returns the obstacle states at the last time step."""
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


class ObstacleIds[K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> IndexArray[Dims[K]]:
        """Returns the obstacle IDs as a NumPy array."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class ObstacleIdAssignment[ObstacleStatesForTimeStepT, IdT, HistoryT](Protocol):
    def __call__(
        self, states: ObstacleStatesForTimeStepT, /, *, history: HistoryT, ids: IdT
    ) -> IdT | None:
        """Returns the IDs to be assigned to the given obstacle states. If no IDs are returned,
        the obstacle states are assumed to be already ordered by their IDs.

        Args:
            states: The obstacle states at the current time step.
            history: The full history of obstacle states.
            ids: The obstacle IDs corresponding to the historical states.
        """
        ...


class ObstacleStatesRunningHistory[ObstacleStatesForTimeStepT, IdT, HistoryT](Protocol):
    def get(self) -> HistoryT:
        """Returns the current full running history of obstacle states."""
        ...

    def ids(self) -> IdT:
        """Returns the current obstacle IDs corresponding to the running history."""
        ...

    def append(self, states: ObstacleStatesForTimeStepT, /, *, ids: IdT | None) -> Self:
        """Appends new obstacle states at a time step to the running history. If the IDs are
        provided, the states must be reordered to match the IDs before appending.
        """
        ...


class ObstacleStateEstimator[HistoryT, StatesT, InputsT, CovarianceT = None](Protocol):
    def estimate_from(
        self, history: HistoryT
    ) -> EstimatedObstacleStates[StatesT, InputsT, CovarianceT]:
        """Estimates the current states and inputs of obstacles given their history."""
        ...


class InputAssumptionProvider[InputsT](Protocol):
    def __call__(self, inputs: InputsT, /) -> InputsT:
        """Applies assumptions to the given inputs and returns the modified inputs."""
        ...


class ObstacleModel[HistoryT, StatesT, InputsT, CovariancesT, StateSequencesT](
    Protocol
):
    def forward(
        self,
        *,
        states: StatesT,
        inputs: InputsT,
        covariances: CovariancesT | None,
        horizon: int,
    ) -> StateSequencesT:
        """Simulates the objects forward in time given the current states, assumed inputs, and horizon.

        Args:
            states: The current states of obstacles.
            inputs: The assumed control inputs.
            covariances: Uncertainty associated with the current states and inputs (if available).
            horizon: The number of time steps to simulate.
        """
        ...


class PredictionCreator[StateSequencesT, PredictionT](Protocol):
    def __call__(self, *, states: StateSequencesT) -> PredictionT:
        """Creates a prediction from the given state sequences."""
        ...

    def empty(self, *, horizon: int) -> PredictionT:
        """Creates an empty obstacle state prediction with the specified horizon."""
        ...


class ObstacleMotionPredictor[HistoryT, PredictionT](Protocol):
    def predict(self, *, history: HistoryT) -> PredictionT:
        """Predicts the future states of obstacles based on their state history."""
        ...
