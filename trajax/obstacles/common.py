from dataclasses import dataclass

from trajax.types import (
    ObstacleStateProvider,
    ObstacleMotionPredictor,
    ObstacleStatesRunningHistory,
)


@dataclass(kw_only=True)
class PredictingObstacleStateProvider[
    ObstacleStatesForTimeStepT,
    HistoryT,
    PredictionT,
](ObstacleStateProvider[PredictionT]):
    predictor: ObstacleMotionPredictor[HistoryT, PredictionT]
    history: ObstacleStatesRunningHistory[ObstacleStatesForTimeStepT, HistoryT]

    @staticmethod
    def create[O, H, P](
        *,
        predictor: ObstacleMotionPredictor[H, P],
        history: ObstacleStatesRunningHistory[O, H],
    ) -> "PredictingObstacleStateProvider[O, H, P]":
        return PredictingObstacleStateProvider(predictor=predictor, history=history)

    def __call__(self) -> PredictionT:
        return self.predictor.predict(history=self.history.get())

    def observe(self, states: ObstacleStatesForTimeStepT) -> None:
        self.history = self.history.append(states)
