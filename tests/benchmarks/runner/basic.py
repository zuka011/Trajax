from dataclasses import dataclass

from tests.benchmarks.runner.common import BenchmarkTarget


@dataclass(frozen=True)
class NumpyBenchmarkRunner:
    @staticmethod
    def create() -> "NumpyBenchmarkRunner":
        """Creates a benchmark runner for NumPy implementations."""
        return NumpyBenchmarkRunner()

    async def warm_up(self, target: BenchmarkTarget) -> None:
        await target()

    async def execute[T](self, target: BenchmarkTarget[T]) -> T:
        return await target()
