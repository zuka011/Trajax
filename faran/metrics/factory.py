from typing import Final

from faran.metrics.registry import MetricRegistry
from faran.metrics.collision import CollisionMetric
from faran.metrics.error import MpccErrorMetric
from faran.metrics.task import TaskCompletionMetric
from faran.metrics.constraint import ConstraintViolationMetric
from faran.metrics.comfort import ComfortMetric


class metrics:
    """Factory namespace for creating simulation evaluation metrics."""

    registry: Final = MetricRegistry.of
    collision: Final = CollisionMetric.create
    mpcc_error: Final = MpccErrorMetric.create
    task_completion: Final = TaskCompletionMetric.create
    constraint_violation: Final = ConstraintViolationMetric.create
    comfort: Final = ComfortMetric.create
