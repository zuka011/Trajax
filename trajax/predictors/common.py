from dataclasses import dataclass

from trajax.types import ObstacleStatesHistory, ObstacleModel, EmptyPredictionCreator


@dataclass(kw_only=True, frozen=True)
class ConstantVelocityPredictor[
    HistoryT: ObstacleStatesHistory,
    StatesT,
    VelocitiesT,
    InputSequencesT,
    PredictionT,
]:
    horizon: int
    model: ObstacleModel[HistoryT, StatesT, VelocitiesT, InputSequencesT, PredictionT]
    empty_prediction: EmptyPredictionCreator[PredictionT]

    @staticmethod
    def create[H: ObstacleStatesHistory, S, V, I, P](
        *,
        horizon: int,
        model: ObstacleModel[H, S, V, I, P],
        empty_prediction: EmptyPredictionCreator[P],
    ) -> "ConstantVelocityPredictor[H, S, V, I, P]":
        return ConstantVelocityPredictor(
            horizon=horizon, model=model, empty_prediction=empty_prediction
        )

    def predict(self, *, history: HistoryT) -> PredictionT:
        if history.horizon == 0:
            return self.empty_prediction(horizon=self.horizon)

        estimated = self.model.estimate_state_from(history)
        input = self.model.input_to_maintain(
            estimated.velocities, states=estimated.states, horizon=self.horizon
        )

        return self.model.forward(current=estimated.states, input=input)
