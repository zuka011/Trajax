from typing import Generator

import jax

from pytest import fixture


@fixture(scope="session", autouse=True)
def log_jax_compilation() -> Generator[None, None, None]:
    """Enables JAX compilation logging for the duration of the test session."""
    jax.config.update("jax_log_compiles", True)
    yield
    jax.config.update("jax_log_compiles", False)
