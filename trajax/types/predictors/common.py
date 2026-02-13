from typing import Protocol, Self, Any, runtime_checkable
from dataclasses import dataclass

from trajax.types.array import DataType

from numtypes import Array, IndexArray, Dims


@dataclass(kw_only=True, frozen=True)
class EstimatedObstacleStates[StatesT, InputsT]:
    states: StatesT
    inputs: InputsT


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


# TODO: Check if this can be removed and replaced with ObstacleStates.
class ObstacleStateSequences[T: int, D_o: int, K: int, SingleSampleT = Any](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, K]]:
        """Returns the obstacle state sequences as a NumPy array."""
        ...

    def single(self) -> SingleSampleT:
        """Returns the state sequences as a sampled obstacle states with one sample."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon of the obstacle state sequences."""
        ...

    @property
    def dimension(self) -> D_o:
        """The dimension of the obstacle states."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class CovarianceSequences[T: int, D_o: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_o, D_o, K]]:
        """Returns the covariance sequences as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon of the covariance sequences."""
        ...

    @property
    def dimension(self) -> D_o:
        """The dimension of the covariance matrices."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class ObstacleControlInputSequences[T: int, D_u: int, K: int](Protocol):
    def __array__(self, dtype: DataType | None = None) -> Array[Dims[T, D_u, K]]:
        """Returns the control input sequences as a NumPy array."""
        ...

    @property
    def horizon(self) -> T:
        """The time horizon of the control input sequences."""
        ...

    @property
    def dimension(self) -> D_u:
        """The dimension of the control inputs."""
        ...

    @property
    def count(self) -> K:
        """The number of obstacles."""
        ...


class ObstacleModel[
    HistoryT,
    StatesT,
    InputsT,
    InputSequencesT,
    StateSequencesT,
    JacobianT = Any,
](Protocol):
    def estimate_state_from(
        self, history: HistoryT
    ) -> EstimatedObstacleStates[StatesT, InputsT]:
        """Estimates the current states and inputs of objects given their history."""
        ...

    def input_to_maintain(
        self, inputs: InputsT, *, states: StatesT, horizon: int
    ) -> InputSequencesT:
        """Generates control input sequences that maintain the given inputs over the specified horizon."""
        ...

    def forward(self, *, current: StatesT, inputs: InputSequencesT) -> StateSequencesT:
        """Simulates the objects forward in time given the current states and control inputs."""
        ...

    # TODO: Review!
    def state_jacobian(
        self, *, states: StateSequencesT, inputs: InputSequencesT
    ) -> JacobianT:
        """Computes the state Jacobian F = ∂f/∂x of the model's forward function.

        This describes how state uncertainty propagates through the dynamics.
        """
        ...

    # TODO: Review!
    def input_jacobian(
        self, *, states: StateSequencesT, inputs: InputSequencesT
    ) -> JacobianT:
        """Computes the input Jacobian G = ∂f/∂u of the model's forward function.

        This describes how input uncertainty enters the state dynamics.
        """
        ...


class PredictionCreator[StateSequencesT, CovarianceSequencesT, PredictionT](Protocol):
    def __call__(
        self, *, states: StateSequencesT, covariances: CovarianceSequencesT
    ) -> PredictionT:
        """Creates a prediction from the given state sequences."""
        ...

    def empty(self, *, horizon: int) -> PredictionT:
        """Creates an empty obstacle state prediction with the specified horizon."""
        ...


class InputAssumptionProvider[InputsT](Protocol):
    def __call__(self, inputs: InputsT, /) -> InputsT:
        """Applies assumptions to the given inputs and returns the modified inputs."""
        ...


class CovariancePropagator[StateSequencesT, CovarianceSequencesT](Protocol):
    def propagate(
        self, *, states: StateSequencesT, inputs: Any = None
    ) -> CovarianceSequencesT:
        """Propagates the covariance of obstacle states over time.

        Args:
            states: The predicted obstacle state sequences.
            inputs: Optional control input sequences, required by state-dependent
                propagators such as EKF.
        """
        ...


# NOTE: Simplifies JAX type checking.
@runtime_checkable
class CovarianceExtractor[InputCovarianceT, OutputCovarianceT](Protocol):
    def __call__(self, covariance: InputCovarianceT, /) -> OutputCovarianceT:
        """Extracts parts of the input covariance to produce the output covariance.

        For example, an extractor may take only the position part of a full state covariance."""
        ...


class ObstacleMotionPredictor[HistoryT, PredictionT](Protocol):
    def predict(self, *, history: HistoryT) -> PredictionT:
        """Predicts the future states of obstacles based on their state history."""
        ...
