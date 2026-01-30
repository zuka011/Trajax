from typing import Any
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkStats:
    """Raw benchmark statistics from pytest-benchmark."""

    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    stddev: float = 0.0
    rounds: int = 0
    median: float = 0.0
    ops: float = 0.0
    iqr: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    iqr_outliers: int = 0
    stddev_outliers: int = 0
    outliers: str = ""
    ld15iqr: float = 0.0
    hd15iqr: float = 0.0
    total: float = 0.0
    data: list[float] = field(default_factory=list)
    iterations: int = 1


@dataclass(frozen=True)
class BenchmarkParams:
    """Benchmark parameter ID from pytest parametrize."""

    id: str = ""


@dataclass(frozen=True)
class Benchmark:
    """Single benchmark result from pytest-benchmark JSON."""

    name: str = ""
    stats: BenchmarkStats = field(default_factory=BenchmarkStats)
    fullname: str = ""
    group: str | None = None
    param: str = ""
    params: BenchmarkParams | None = None
    extra_info: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MachineInfo:
    """Machine information from benchmark environment."""

    python_version: str = ""
    platform: str = ""
    cpu: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitInfo:
    """Git commit information."""

    id: str = ""


@dataclass(frozen=True)
class BenchmarkData:
    """Complete benchmark data from pytest-benchmark JSON file."""

    machine_info: MachineInfo | None = None
    commit_info: CommitInfo | None = None
    benchmarks: list[Benchmark] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedBenchmark:
    """Processed benchmark data for export and analysis."""

    name: str
    group: str
    implementation: str
    parameters: dict[str, Any]
    mean: float
    min: float
    max: float
    stddev: float
    median: float
    rounds: int
    ops: float
    iqr: float


@dataclass
class ExportData:
    """Complete export structure for JSON output."""

    machine_info: dict[str, Any]
    commit_info: dict[str, Any]
    benchmarks: list[ParsedBenchmark]
    summary: dict[str, Any]
