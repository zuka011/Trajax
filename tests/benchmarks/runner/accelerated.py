from typing import Literal
from dataclasses import dataclass
from contextlib import contextmanager

import jax

from tests.benchmarks.runner.common import BenchmarkTarget

type Device = Literal["CPU", "GPU"]


@dataclass(frozen=True)
class JaxBenchmarkRunner:
    device: Device

    @staticmethod
    def create(*, device: Device | None = None) -> "JaxBenchmarkRunner":
        """Creates a benchmark runner for JAX implementations."""
        if device is None:
            device = detect_device()

        return JaxBenchmarkRunner(device=device)

    @staticmethod
    def supports(*, device: Device) -> bool:
        """Checks whether the given device is supported."""
        return check_device(device)

    def __repr__(self) -> str:
        return self.name

    def warm_up(self, target: BenchmarkTarget) -> None:
        with self._on_device():
            result = target()
            jax.block_until_ready(result)

    def execute[T](self, target: BenchmarkTarget[T]) -> T:
        with self._on_device():
            result = target()
            jax.block_until_ready(result)
            return result

    def is_slow_for(self, *, risk_metric_sample_count: int) -> bool:
        return False

    @property
    def name(self) -> str:
        return f"JAX ({self.device})"

    @contextmanager
    def _on_device(self):
        devices = jax.devices(self.device.lower())

        with jax.default_device(devices[0]):
            yield


def detect_device() -> Device:
    """Detect the best available device (prefers GPU)."""
    return "GPU" if check_device("GPU") else "CPU"


def check_device(device: Device) -> bool:
    """Check if the specified device is available."""
    try:
        jax.devices(device.lower())
        return True
    except RuntimeError:
        return False
