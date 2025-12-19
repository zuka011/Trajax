from dataclasses import dataclass

import jax

from tests.benchmarks.runner.common import BenchmarkTarget


@dataclass(frozen=True)
class JaxBenchmarkRunner:
    @staticmethod
    def create() -> "JaxBenchmarkRunner":
        """Creates a benchmark runner for JAX implementations."""
        return JaxBenchmarkRunner()

    async def warm_up(self, target: BenchmarkTarget) -> None:
        result = await target()
        jax.block_until_ready(result)

    async def execute[T](self, target: BenchmarkTarget[T]) -> T:
        result = await target()
        jax.block_until_ready(result)
        return result
