from typing import Any
from dataclasses import dataclass

from trajax.types import (
    ObstacleStatesForTimeStep,
    ObstacleStateProvider,
    ObstacleMotionPredictor,
    ObstacleStatesRunningHistory,
    ObstacleIdAssignment,
)


class DefaultIdAssignment(ObstacleIdAssignment[Any, Any, Any]):
    def __call__(self, states: Any, /, *, history: Any, ids: Any) -> None:
        return None


@dataclass(kw_only=True)
class PredictingObstacleStateProvider[
    ObstacleStatesForTimeStepT,
    IdT,
    HistoryT,
    PredictionT,
](ObstacleStateProvider[PredictionT]):
    predictor: ObstacleMotionPredictor[HistoryT, PredictionT]
    history: ObstacleStatesRunningHistory[ObstacleStatesForTimeStepT, IdT, HistoryT]
    id_assignment: ObstacleIdAssignment[ObstacleStatesForTimeStepT, IdT, HistoryT]

    @staticmethod
    def create[O: ObstacleStatesForTimeStep, I, H, P](
        *,
        predictor: ObstacleMotionPredictor[H, P],
        history: ObstacleStatesRunningHistory[O, I, H],
        id_assignment: ObstacleIdAssignment[O, I, H] | None = None,
    ) -> "PredictingObstacleStateProvider[O, I, H, P]":
        return PredictingObstacleStateProvider(
            predictor=predictor,
            history=history,
            id_assignment=id_assignment
            if id_assignment is not None
            else DefaultIdAssignment(),
        )

    def __call__(self) -> PredictionT:
        return self.predictor.predict(history=self.history.get())

    def observe(self, states: ObstacleStatesForTimeStepT) -> None:
        self.history = self.history.append(
            states,
            ids=self.id_assignment(
                states, history=self.history.get(), ids=self.history.ids()
            ),
        )
