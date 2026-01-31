import asyncio
from typing import Generator, MutableMapping

import jax

from tests.benchmarks.runner import gpu_info
from pytest import fixture, Config


def pytest_configure(config) -> None:
    jax.config.update("jax_enable_x64", True)


def pytest_benchmark_update_machine_info(
    config: Config, machine_info: MutableMapping[str, object]
):
    machine_info["gpu"] = asyncio.run(gpu_info())


@fixture(scope="session", autouse=True)
def verify_jax_double_precision() -> Generator[None, None, None]:
    """Verifies that JAX double precision is enabled for the duration of the test session."""
    assert jax.config.read("jax_enable_x64"), "JAX double precision is not enabled."
    yield
    assert jax.config.read("jax_enable_x64"), (
        "JAX double precision was disabled during the test session."
    )


@fixture(scope="session", autouse=True)
def log_jax_compilation() -> Generator[None, None, None]:
    """Enables JAX compilation logging for the duration of the test session."""
    jax.config.update("jax_log_compiles", True)
    yield
    jax.config.update("jax_log_compiles", False)
