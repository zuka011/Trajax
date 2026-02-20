from typing import Final

from faran.collectors.registry import CollectorRegistry
from faran.collectors.obstacles import (
    ObstacleStateCollector,
    ObstacleForecastCollector,
)
from faran.collectors.risk import RiskCollector
from faran.collectors.state import (
    StateCollector,
    ControlCollector,
    TrajectoryCollector,
)


class collectors:
    """Factory namespace for creating simulation data collectors."""

    states: Final = StateCollector
    controls: Final = ControlCollector
    trajectories: Final = TrajectoryCollector
    risk: Final = RiskCollector
    obstacle_states: Final = ObstacleStateCollector
    obstacle_forecasts: Final = ObstacleForecastCollector
    registry: Final = CollectorRegistry.of
