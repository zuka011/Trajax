from dataclasses import dataclass

from trajax.types import ObstacleStatesHistory, ObstacleModel, PredictionCreator


@dataclass(kw_only=True, frozen=True)
class CurvilinearPredictor[
    HistoryT: ObstacleStatesHistory,
    StatesT,
    VelocitiesT,
    InputSequencesT,
    StateSequencesT,
    PredictionT,
]:
    horizon: int
    model: ObstacleModel[
        HistoryT, StatesT, VelocitiesT, InputSequencesT, StateSequencesT
    ]
    prediction: PredictionCreator[StateSequencesT, PredictionT]

    @staticmethod
    def create[H: ObstacleStatesHistory, S, V, IS, SS, P](
        *,
        horizon: int,
        model: ObstacleModel[H, S, V, IS, SS],
        prediction: PredictionCreator[SS, P],
    ) -> "CurvilinearPredictor[H, S, V, IS, SS, P]":
        return CurvilinearPredictor(horizon=horizon, model=model, prediction=prediction)

    def predict(self, *, history: HistoryT) -> PredictionT:
        if history.horizon == 0:
            return self.prediction.empty(horizon=self.horizon)

        estimated = self.model.estimate_state_from(history)
        inputs = self.model.input_to_maintain(
            estimated.velocities, states=estimated.states, horizon=self.horizon
        )
        states = self.model.forward(current=estimated.states, inputs=inputs)

        return self.prediction(states=states)
