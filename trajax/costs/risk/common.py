from typing import Protocol
from dataclasses import dataclass


class RiskMetric(Protocol):
    @property
    def sample_count(self) -> int:
        """Returns the number of samples used in the risk metric computation."""
        ...


class NoMetric:
    @staticmethod
    def create() -> "NoMetric":
        return NoMetric()

    @property
    def sample_count(self) -> int:
        return 1


@dataclass(frozen=True)
class MeanVarianceMetric:
    gamma: float
    sample_count: int

    @staticmethod
    def create(*, gamma: float, sample_count: int) -> "MeanVarianceMetric":
        return MeanVarianceMetric(gamma=gamma, sample_count=sample_count)
