from typing import Any, cast
from dataclasses import dataclass

from trajax.types import (
    ObstacleStatesForTimeStep,
    ObstacleStatesHistory,
    CovariancePropagator,
    ObstacleModel,
    PredictionCreator,
)


class NoCovariance:
    def propagate(self, *args, **kwargs) -> None:
        return None


@dataclass(kw_only=True, frozen=True)
class CovariancePadding:
    to_dimension: int
    epsilon: float

    @staticmethod
    def create(*, to_dimension: int, epsilon: float) -> "CovariancePadding":
        return CovariancePadding(to_dimension=to_dimension, epsilon=epsilon)

    def __post_init__(self):
        assert self.to_dimension > 0, (
            f"Covariance target dimension must be positive, got {self.to_dimension}."
        )
        assert self.epsilon > 0, (
            f"Covariance padding epsilon must be positive, got {self.epsilon}."
        )


@dataclass(kw_only=True, frozen=True)
class StaticPredictor:
    horizon: int

    @staticmethod
    def create(*, horizon: int) -> "StaticPredictor ":
        return StaticPredictor(horizon=horizon)

    def predict[PredictionT](
        self,
        *,
        history: ObstacleStatesHistory[
            Any, Any, Any, ObstacleStatesForTimeStep[Any, PredictionT]
        ],
    ) -> PredictionT:
        assert history.horizon > 0, (
            "There should be at least one observation in history."
        )

        return history.last().replicate(horizon=self.horizon)


@dataclass(kw_only=True, frozen=True)
class CurvilinearPredictor[
    HistoryT: ObstacleStatesHistory,
    StatesT,
    VelocitiesT,
    InputSequencesT,
    StateSequencesT,
    CovarianceSequencesT,
    PredictionT,
]:
    horizon: int
    model: ObstacleModel[
        HistoryT, StatesT, VelocitiesT, InputSequencesT, StateSequencesT
    ]
    propagator: CovariancePropagator[StateSequencesT, CovarianceSequencesT]
    prediction: PredictionCreator[StateSequencesT, CovarianceSequencesT, PredictionT]

    @staticmethod
    def create[H: ObstacleStatesHistory, S, V, IS, SS, CS, P](
        *,
        horizon: int,
        model: ObstacleModel[H, S, V, IS, SS],
        prediction: PredictionCreator[SS, CS, P],
        propagator: CovariancePropagator[SS, CS] | None = None,
    ) -> "CurvilinearPredictor[H, S, V, IS, SS, CS, P]":
        return CurvilinearPredictor(
            horizon=horizon,
            model=model,
            prediction=prediction,
            propagator=propagator
            if propagator is not None
            else cast(CovariancePropagator[SS, CS], NoCovariance()),
        )

    def predict(self, *, history: HistoryT) -> PredictionT:
        if history.horizon == 0:
            return self.prediction.empty(horizon=self.horizon)

        estimated = self.model.estimate_state_from(history)
        inputs = self.model.input_to_maintain(
            estimated.velocities, states=estimated.states, horizon=self.horizon
        )
        states = self.model.forward(current=estimated.states, inputs=inputs)
        covariances = self.propagator.propagate(states=states)

        return self.prediction(states=states, covariances=covariances)
