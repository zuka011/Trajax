from dataclasses import dataclass

from tests.benchmarks.runner.common import BenchmarkTarget


@dataclass(frozen=True)
class NumPyBenchmarkRunner:
    @staticmethod
    def create() -> "NumPyBenchmarkRunner":
        """Creates a benchmark runner for NumPy implementations."""
        return NumPyBenchmarkRunner()

    def __repr__(self) -> str:
        return self.name

    def warm_up(self, target: BenchmarkTarget) -> None:
        target()

    def execute[T](self, target: BenchmarkTarget[T]) -> T:
        return target()

    def is_slow_for(self, *, risk_metric_sample_count: int) -> bool:
        return risk_metric_sample_count > 100

    @property
    def name(self) -> str:
        return "NumPy"
