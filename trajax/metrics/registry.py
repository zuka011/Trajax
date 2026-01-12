from typing import Mapping, Any, overload
from dataclasses import dataclass

from trajax.types import SimulationData, Metric

from trajax.collectors import CollectorRegistry


@dataclass(frozen=True)
class MetricRegistry:
    metrics_by_name: Mapping[str, Metric]
    collectors: CollectorRegistry

    @staticmethod
    def of(*metrics: Metric, collectors: CollectorRegistry) -> "MetricRegistry":
        return MetricRegistry(
            metrics_by_name={metric.name: metric for metric in metrics},
            collectors=collectors,
        )

    @overload
    def get[R](self, metric: Metric[R]) -> R:
        """Retrieves results for the specified metric from the registry."""
        ...

    @overload
    def get(self, metric: str) -> Any:
        """Retrieves results for the specified metric from the registry."""
        ...

    def get[R](self, metric: Metric[R] | str) -> R | Any:
        name = metric if isinstance(metric, str) else metric.name
        found = self.metrics_by_name.get(name)

        assert found is not None, f"Metric with name '{name}' not found in registry."

        return found.compute(self.data)

    @property
    def data(self) -> SimulationData:
        """Retrieves the collected simulation data."""
        return self.collectors.data
