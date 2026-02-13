from typing import Any, cast
from dataclasses import dataclass

from trajax.types import (
    ObstacleStatesForTimeStep,
    ObstacleStatesHistory,
    CovariancePropagator,
    CovarianceExtractor,
    ObstacleModel,
    ObstacleStateEstimator,
    PredictionCreator,
    InputAssumptionProvider,
)


class NoCovariance:
    """Stub covariance propagator that returns None, indicating no uncertainty."""

    def propagate(self, *args, **kwargs) -> None:
        return None


class KeepFullCovariance:
    """Covariance extractor that keeps the full covariance matrix unchanged."""

    def __call__(self, covariance: Any, /) -> Any:
        return covariance


@dataclass(kw_only=True, frozen=True)
class CovarianceResizing[InputCovarianceT, OutputCovarianceT]:
    """Extracts state dimensions and pads a covariance matrix to a target dimension with a small epsilon diagonal."""

    keep: CovarianceExtractor[InputCovarianceT, OutputCovarianceT]
    pad_to: int | None
    epsilon: float

    @staticmethod
    def identity() -> "CovarianceResizing[Any, Any]":
        """Returns a covariance resizing that leaves the covariance unchanged."""
        return CovarianceResizing.create()

    @staticmethod
    def create[I, O](
        *,
        keep: CovarianceExtractor[I, O] | None = None,
        pad_to: int | None = None,
        epsilon: float = 1e-13,
    ) -> "CovarianceResizing[I, O]":
        """Creates a covariance resizing description.

        A covariance resizing consists of two parts:
        1. An extractor that extracts the relevant covariance submatrix.
        2. A padding specification that pads the extracted covariance to the desired dimension.

        Both of the parts are optional, so a resizing can also leave the covariance unchanged.
        """
        return CovarianceResizing(
            keep=keep or cast(CovarianceExtractor[I, O], KeepFullCovariance()),
            pad_to=pad_to,
            epsilon=epsilon,
        )

    def __post_init__(self) -> None:
        assert self.pad_to is None or self.pad_to > 0, (
            f"Covariance target dimension after padding must be positive, got {self.pad_to}."
        )
        assert self.epsilon > 0, (
            f"Covariance padding epsilon must be positive, got {self.epsilon}."
        )


@dataclass(kw_only=True, frozen=True)
class StaticPredictor:
    """Predicts obstacle motion by replicating the last observed state over the horizon."""

    horizon: int

    @staticmethod
    def create(*, horizon: int) -> "StaticPredictor":
        return StaticPredictor(horizon=horizon)

    def predict[PredictionT](
        self,
        *,
        history: ObstacleStatesHistory[
            Any, Any, Any, ObstacleStatesForTimeStep[Any, Any, PredictionT]
        ],
    ) -> PredictionT:
        assert history.horizon > 0, (
            "There should be at least one observation in history."
        )

        return history.last().replicate(horizon=self.horizon)


class NoAssumptions[InputsT]:
    """Identity assumption provider that passes inputs through unchanged."""

    def __call__(self, inputs: InputsT, /) -> InputsT:
        return inputs


@dataclass(kw_only=True, frozen=True)
class CurvilinearPredictor[
    HistoryT: ObstacleStatesHistory,
    StatesT,
    InputsT,
    StateSequencesT,
    CovarianceSequencesT,
    PredictionT,
]:
    """Predicts obstacle motion by estimating inputs and propagating forward with a dynamical model."""

    horizon: int
    model: ObstacleModel[HistoryT, StatesT, InputsT, StateSequencesT]
    estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT]
    propagator: CovariancePropagator[StateSequencesT, CovarianceSequencesT]
    prediction: PredictionCreator[StateSequencesT, CovarianceSequencesT, PredictionT]
    assumptions: InputAssumptionProvider[InputsT]

    @staticmethod
    def create[H: ObstacleStatesHistory, S, I, SS, CS, P](
        *,
        horizon: int,
        model: ObstacleModel[H, S, I, SS],
        estimator: ObstacleStateEstimator[H, S, I],
        prediction: PredictionCreator[SS, CS, P],
        propagator: CovariancePropagator[SS, CS] | None = None,
        assumptions: InputAssumptionProvider[I] | None = None,
    ) -> "CurvilinearPredictor[H, S, I, SS, CS, P]":
        return CurvilinearPredictor(
            horizon=horizon,
            model=model,
            estimator=estimator,
            prediction=prediction,
            propagator=propagator
            if propagator is not None
            else cast(CovariancePropagator[SS, CS], NoCovariance()),
            assumptions=assumptions
            if assumptions is not None
            else cast(InputAssumptionProvider[I], NoAssumptions()),
        )

    def predict(self, *, history: HistoryT) -> PredictionT:
        if history.horizon == 0:
            return self.prediction.empty(horizon=self.horizon)

        estimated = self.estimator.estimate_from(history)
        inputs = self.assumptions(estimated.inputs)
        states = self.model.forward(
            current=estimated.states, inputs=inputs, horizon=self.horizon
        )
        covariances = self.propagator.propagate(states=states, inputs=inputs)

        return self.prediction(states=states, covariances=covariances)
