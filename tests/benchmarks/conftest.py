from typing import Generator

import jax

from pytest import fixture


def pytest_configure(config) -> None:
    jax.config.update("jax_enable_x64", True)


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
