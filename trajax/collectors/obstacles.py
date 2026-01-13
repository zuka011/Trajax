from dataclasses import dataclass, field

from trajax.types import ObstacleStateObserver, ObstacleStateProvider
from trajax.collectors.access import access
from trajax.collectors.registry import (
    DataTransformer,
    IdentityTransformer,
    ModificationNotifierMixin,
    ListCollectorMixin,
    OnModifyCallback,
)


@dataclass(frozen=True)
class ObstacleStateCollector[ObstacleStatesForTimeStepT](
    ObstacleStateObserver[ObstacleStatesForTimeStepT],
    ModificationNotifierMixin,
    ListCollectorMixin,
):
    inner: ObstacleStateObserver[ObstacleStatesForTimeStepT]
    transformer: DataTransformer[ObstacleStatesForTimeStepT]
    _callbacks: list[OnModifyCallback] = field(default_factory=list)
    _collected: list[ObstacleStatesForTimeStepT] = field(default_factory=list)

    @staticmethod
    def decorating[OS](
        observer: ObstacleStateObserver[OS],
        *,
        transformer: DataTransformer[OS] = IdentityTransformer(),
    ) -> "ObstacleStateCollector[OS]":
        return ObstacleStateCollector(observer, transformer=transformer)

    def observe(self, states: ObstacleStatesForTimeStepT) -> None:
        self._collected.append(states)
        self.inner.observe(states)
        self.notify()

    @property
    def key(self) -> str:
        return access.obstacle_states.key


@dataclass(frozen=True)
class ObstacleForecastCollector[ObstacleStatesT](
    ObstacleStateProvider[ObstacleStatesT],
    ModificationNotifierMixin,
    ListCollectorMixin,
):
    inner: ObstacleStateProvider[ObstacleStatesT]
    transformer: DataTransformer[ObstacleStatesT]
    _callbacks: list[OnModifyCallback] = field(default_factory=list)
    _collected: list[ObstacleStatesT] = field(default_factory=list)

    @staticmethod
    def decorating[OS](
        provider: ObstacleStateProvider[OS],
        *,
        transformer: DataTransformer[OS] = IdentityTransformer(),
    ) -> "ObstacleForecastCollector[OS]":
        return ObstacleForecastCollector(provider, transformer=transformer)

    def __call__(self) -> ObstacleStatesT:
        states = self.inner()
        self._collected.append(states)
        self.notify()
        return states

    @property
    def key(self) -> str:
        return access.obstacle_forecasts.key
