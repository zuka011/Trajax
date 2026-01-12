from typing import Sequence
from dataclasses import dataclass, field

from trajax.types import ObstacleStateObserver, ObstacleStateProvider
from trajax.collectors.access import access


@dataclass(frozen=True)
class ObstacleStateCollector[ObstacleStatesForTimeStepT](
    ObstacleStateObserver[ObstacleStatesForTimeStepT]
):
    inner: ObstacleStateObserver[ObstacleStatesForTimeStepT]
    _collected: list[ObstacleStatesForTimeStepT] = field(default_factory=list)

    @staticmethod
    def decorating[OS](
        observer: ObstacleStateObserver[OS],
    ) -> "ObstacleStateCollector[OS]":
        return ObstacleStateCollector(observer)

    def observe(self, states: ObstacleStatesForTimeStepT) -> None:
        self._collected.append(states)
        self.inner.observe(states)

    @property
    def collected(self) -> Sequence[ObstacleStatesForTimeStepT]:
        return self._collected

    @property
    def key(self) -> str:
        return access.obstacle_states.key


@dataclass(frozen=True)
class ObstacleForecastCollector[
    ObstacleStatesT,
](ObstacleStateProvider[ObstacleStatesT]):
    inner: ObstacleStateProvider[ObstacleStatesT]
    _collected: list[ObstacleStatesT] = field(default_factory=list)

    @staticmethod
    def decorating[OS](
        provider: ObstacleStateProvider[OS],
    ) -> "ObstacleForecastCollector[OS]":
        return ObstacleForecastCollector(provider)

    def __call__(self) -> ObstacleStatesT:
        states = self.inner()
        self._collected.append(states)
        return states

    @property
    def collected(self) -> Sequence[ObstacleStatesT]:
        return self._collected

    @property
    def key(self) -> str:
        return access.obstacle_forecasts.key
