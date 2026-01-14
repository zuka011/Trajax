from typing import Mapping, Any, overload
from dataclasses import dataclass

from trajax.types import SimulationData, Metric

from trajax.collectors import CollectorRegistry


@dataclass
class MetricRegistry:
    metrics_by_name: Mapping[str, Metric]
    computed_metrics_by_name: dict[str, Any]
    collectors: CollectorRegistry

    @staticmethod
    def of(*metrics: Metric, collectors: CollectorRegistry) -> "MetricRegistry":
        return MetricRegistry(
            metrics_by_name={metric.name: metric for metric in metrics},
            computed_metrics_by_name={},
            collectors=collectors,
        )

    def __post_init__(self) -> None:
        self.collectors.on_modified(self._modified)

    @overload
    def get[R](self, metric: Metric[R]) -> R:
        """Retrieves results for the specified metric from the registry."""
        ...

    @overload
    def get(self, metric: str) -> Any:
        """Retrieves results for the specified metric from the registry."""
        ...

    def get[R](self, metric: Metric[R] | str) -> R | Any:
        if (
            name := metric if isinstance(metric, str) else metric.name
        ) not in self.computed_metrics_by_name:
            self.computed_metrics_by_name[name] = self._compute_metric(name)

        return self.computed_metrics_by_name[name]

    @property
    def data(self) -> SimulationData:
        """Retrieves the collected simulation data."""
        return self.collectors.data

    def _modified(self) -> None:
        self.computed_metrics_by_name = {}

    def _compute_metric(self, name: str) -> Any:
        found_metric = self.metrics_by_name.get(name)

        assert found_metric is not None, (
            f"Metric with name '{name}' not found in registry."
        )

        return found_metric.compute(self.data)
