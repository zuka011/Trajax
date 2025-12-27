from dataclasses import dataclass

from tests.benchmarks.runner.common import BenchmarkTarget


@dataclass(frozen=True)
class NumPyBenchmarkRunner:
    @staticmethod
    def create() -> "NumPyBenchmarkRunner":
        """Creates a benchmark runner for NumPy implementations."""
        return NumPyBenchmarkRunner()

    def warm_up(self, target: BenchmarkTarget) -> None:
        target()

    def execute[T](self, target: BenchmarkTarget[T]) -> T:
        return target()
