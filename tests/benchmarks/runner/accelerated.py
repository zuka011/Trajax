from dataclasses import dataclass

import jax

from tests.benchmarks.runner.common import BenchmarkTarget


@dataclass(frozen=True)
class JaxBenchmarkRunner:
    @staticmethod
    def create() -> "JaxBenchmarkRunner":
        """Creates a benchmark runner for JAX implementations."""
        return JaxBenchmarkRunner()

    def __repr__(self) -> str:
        return self.name

    def warm_up(self, target: BenchmarkTarget) -> None:
        result = target()
        jax.block_until_ready(result)

    def execute[T](self, target: BenchmarkTarget[T]) -> T:
        result = target()
        jax.block_until_ready(result)
        return result

    def is_slow_for(self, *, risk_metric_sample_count: int) -> bool:
        return False

    @property
    def name(self) -> str:
        return "JAX"
