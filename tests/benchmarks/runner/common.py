import asyncio
from typing import Protocol, Awaitable

from pytest_benchmark.fixture import BenchmarkFixture


class BenchmarkTarget[T](Protocol):
    async def __call__(self) -> T:
        """Runs the asynchronous benchmark target and returns its result."""
        ...


class BenchmarkRunner(Protocol):
    async def warm_up(self, target: BenchmarkTarget) -> None:
        """Performs any necessary warmup for the benchmark target."""
        ...

    async def execute[T](self, target: BenchmarkTarget[T]) -> T:
        """Executes the benchmark target and returns its result."""
        ...


def run_synchronously[T](coroutine: Awaitable[T]) -> T:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coroutine)


def run_benchmark[T](
    benchmark: BenchmarkFixture,
    runner: BenchmarkRunner,
    *,
    target: BenchmarkTarget[T],
    warmup_iterations: int = 3,
) -> T:
    """Run a benchmark with warmup iterations."""
    for _ in range(warmup_iterations):
        run_synchronously(runner.warm_up(target))

    return benchmark(lambda: run_synchronously(runner.execute(target)))
