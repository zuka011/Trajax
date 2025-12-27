from typing import Protocol

from pytest_benchmark.fixture import BenchmarkFixture


class BenchmarkTarget[T](Protocol):
    def __call__(self) -> T:
        """Runs the asynchronous benchmark target and returns its result."""
        ...


class BenchmarkRunner(Protocol):
    def warm_up(self, target: BenchmarkTarget) -> None:
        """Performs any necessary warmup for the benchmark target."""
        ...

    def execute[T](self, target: BenchmarkTarget[T]) -> T:
        """Executes the benchmark target and returns its result."""
        ...


def run_benchmark[T](
    benchmark: BenchmarkFixture,
    runner: BenchmarkRunner,
    *,
    target: BenchmarkTarget[T],
    warmup_iterations: int = 3,
) -> T:
    """Run a benchmark with warmup iterations."""
    for _ in range(warmup_iterations):
        runner.warm_up(target)

    return benchmark(lambda: runner.execute(target))
