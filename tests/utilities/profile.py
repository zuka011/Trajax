import re
import time
import logging
from dataclasses import dataclass, field

import jax

import pytest


@dataclass
class CompilationTracker:
    _compile_time: float = field(default=0.0, init=False)
    _compile_count: int = field(default=0, init=False)
    _start_time: float = field(default=0.0, init=False)
    _handler: logging.Handler | None = field(default=None, init=False)
    _original_handlers: list[logging.Handler] = field(default_factory=list, init=False)

    def start(self) -> None:
        self._compile_time = 0.0
        self._compile_count = 0
        self._start_time = time.perf_counter()

        jax.config.update("jax_log_compiles", True)

        logger = logging.getLogger("jax")
        self._original_handlers = logger.handlers.copy()

        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        self._handler = _CompileLogHandler(self)
        logger.addHandler(self._handler)

    def stop(self) -> str:
        jax.config.update("jax_log_compiles", False)

        logger = logging.getLogger("jax")
        if self._handler:
            logger.removeHandler(self._handler)

        logger.handlers = self._original_handlers
        logger.propagate = True

        total = time.perf_counter() - self._start_time
        pct = (self._compile_time / total) * 100 if total > 0 else 0

        return (
            f"JAX compilation: {self._compile_time:.2f}s / {total:.2f}s ({pct:.1f}%) "
            f"[{self._compile_count} compilations]"
        )


class _CompileLogHandler(logging.Handler):
    _pattern: re.Pattern[str] = re.compile(
        r"Finished XLA compilation.*in ([\d.]+)\s*(ms|s)"
    )

    def __init__(self, tracker: CompilationTracker) -> None:
        super().__init__()
        self._tracker = tracker

    def emit(self, record: logging.LogRecord) -> None:
        if match := self._pattern.search(record.getMessage()):
            duration = float(match.group(1))
            unit = match.group(2)
            self._tracker._compile_time += duration / 1000 if unit == "ms" else duration
            self._tracker._compile_count += 1


compilation_tracker = CompilationTracker()


def add_compilation_tracker_option(parser: pytest.Parser) -> None:
    parser.addoption(
        "--jax-profile",
        action="store_true",
        default=False,
        help="Profile JAX compilation time",
    )


def is_compilation_tracker_enabled(session: pytest.Session) -> bool:
    return session.config.getoption("--jax-profile")
