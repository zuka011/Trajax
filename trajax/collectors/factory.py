from typing import Final

from trajax.collectors.registry import CollectorRegistry
from trajax.collectors.obstacles import (
    ObstacleStateCollector,
    ObstacleForecastCollector,
)
from trajax.collectors.risk import RiskCollector
from trajax.collectors.state import (
    StateCollector,
    ControlCollector,
    TrajectoryCollector,
)


class collectors:
    states: Final = StateCollector
    controls: Final = ControlCollector
    trajectories: Final = TrajectoryCollector
    risk: Final = RiskCollector
    obstacles: Final = ObstacleStateCollector
    obstacle_forecasts: Final = ObstacleForecastCollector
    registry: Final = CollectorRegistry.of
