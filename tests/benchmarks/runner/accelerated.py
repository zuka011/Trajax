from dataclasses import dataclass

import jax

from tests.benchmarks.runner.common import BenchmarkTarget


@dataclass(frozen=True)
class JaxBenchmarkRunner:
    @staticmethod
    def create() -> "JaxBenchmarkRunner":
        """Creates a benchmark runner for JAX implementations."""
        return JaxBenchmarkRunner()

    def warm_up(self, target: BenchmarkTarget) -> None:
        result = target()
        jax.block_until_ready(result)

    def execute[T](self, target: BenchmarkTarget[T]) -> T:
        result = target()
        jax.block_until_ready(result)
        return result
