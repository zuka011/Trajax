from typing import Final

from trajax.metrics.registry import MetricRegistry
from trajax.metrics.collision import CollisionMetric
from trajax.metrics.error import MpccErrorMetric
from trajax.metrics.task import TaskCompletionMetric


class metrics:
    registry: Final = MetricRegistry.of
    collision: Final = CollisionMetric.create
    mpcc_error: Final = MpccErrorMetric.create
    task_completion: Final = TaskCompletionMetric.create
