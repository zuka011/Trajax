from typing import Final

from trajax.metrics.registry import MetricRegistry
from trajax.metrics.collision import CollisionMetric
from trajax.metrics.error import MpccErrorMetric


class metrics:
    registry: Final = MetricRegistry.of
    collision: Final = CollisionMetric.create
    mpcc_error: Final = MpccErrorMetric.create
