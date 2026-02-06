from typing import Final

from trajax.metrics.registry import MetricRegistry
from trajax.metrics.collision import CollisionMetric
from trajax.metrics.error import MpccErrorMetric
from trajax.metrics.task import TaskCompletionMetric
from trajax.metrics.constraint import ConstraintViolationMetric
from trajax.metrics.comfort import ComfortMetric


class metrics:
    """Factory namespace for creating simulation evaluation metrics."""

    registry: Final = MetricRegistry.of
    collision: Final = CollisionMetric.create
    mpcc_error: Final = MpccErrorMetric.create
    task_completion: Final = TaskCompletionMetric.create
    constraint_violation: Final = ConstraintViolationMetric.create
    comfort: Final = ComfortMetric.create
