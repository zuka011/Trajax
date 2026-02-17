from typing import Any, cast
from dataclasses import dataclass

from trajax.types import (
    ObstacleStatesForTimeStep,
    ObstacleStatesHistory,
    ObstacleModel,
    ObstacleStateEstimator,
    PredictionCreator,
    InputAssumptionProvider,
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
    CovarianceT,
    StateSequencesT,
    PredictionT,
]:
    """Predicts obstacle motion by forward propagating estimated states with a curvilinear model."""

    horizon: int
    estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT]
    assumptions: InputAssumptionProvider[InputsT]
    model: ObstacleModel[HistoryT, StatesT, InputsT, CovarianceT, StateSequencesT]
    prediction: PredictionCreator[StateSequencesT, PredictionT]

    @staticmethod
    def create[H: ObstacleStatesHistory, S, I, C, SS, P](
        *,
        horizon: int,
        model: ObstacleModel[H, S, I, C, SS],
        estimator: ObstacleStateEstimator[H, S, I],
        prediction: PredictionCreator[SS, P],
        assumptions: InputAssumptionProvider[I] | None = None,
    ) -> "CurvilinearPredictor[H, S, I, C, SS, P]":
        assumptions = (
            assumptions
            if assumptions is not None
            else cast(InputAssumptionProvider[I], NoAssumptions())
        )

        return CurvilinearPredictor(
            horizon=horizon,
            estimator=estimator,
            assumptions=assumptions,
            model=model,
            prediction=prediction,
        )

    def predict(self, *, history: HistoryT) -> PredictionT:
        if history.horizon == 0:
            return self.prediction.empty(horizon=self.horizon)

        estimated = self.estimator.estimate_from(history)
        inputs = self.assumptions(estimated.inputs)

        return self.prediction(
            states=self.model.forward(
                states=estimated.states,
                inputs=inputs,
                covariances=estimated.covariance,
                horizon=self.horizon,
            )
        )
