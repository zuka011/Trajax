import asyncio
import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiofiles
import msgspec
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    help="Generate pretty benchmark reports and comparisons from pytest-benchmark JSON.",
    no_args_is_help=True,
)
console = Console()


@dataclass(frozen=True)
class MachineInfo:
    python_version: str = ""
    platform: str = ""
    cpu: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommitInfo:
    id: str = ""


@dataclass(frozen=True)
class BenchmarkStats:
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    std: float = 0.0
    rounds: int = 0
    median: float = 0.0
    ops: float = 0.0
    iqr: float = 0.0
    q1: float = 0.0
    q3: float = 0.0
    iqr_outliers: int = 0
    std_outliers: int = 0
    outliers: str = ""
    ld15iqr: float = 0.0
    hd15iqr: float = 0.0
    total: float = 0.0
    data: list[float] = field(default_factory=list)
    iterations: int = 1


@dataclass(frozen=True)
class BenchmarkParams:
    id: str = ""


@dataclass(frozen=True)
class Benchmark:
    name: str = ""
    stats: BenchmarkStats = field(default_factory=BenchmarkStats)
    fullname: str = ""
    group: str = "default"
    param: str = ""
    params: BenchmarkParams | None = None
    extra_info: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkData:
    machine_info: MachineInfo | None = None
    commit_info: CommitInfo | None = None
    benchmarks: list[Benchmark] = field(default_factory=list)


class format:
    @staticmethod
    def time(seconds: float) -> str:
        if seconds < 1e-6:
            return f"{seconds * 1e9:.2f} ns"
        if seconds < 1e-3:
            return f"{seconds * 1e6:.2f} µs"
        if seconds < 1:
            return f"{seconds * 1e3:.2f} ms"
        return f"{seconds:.2f} s"

    @staticmethod
    def operations(ops: float) -> str:
        if ops >= 1_000_000:
            return f"{ops / 1_000_000:.1f}M"
        if ops >= 1_000:
            return f"{ops / 1_000:.1f}K"
        return f"{ops:.1f}"

    @staticmethod
    def comparison(value: float, baseline: float) -> Text:
        if baseline <= 0:
            return Text("N/A", style="dim")
        if (ratio := value / baseline) <= 1.0:
            return Text(f"✓ {ratio:.2f}x", style="green")
        if ratio <= 1.2:
            return Text(f"~ {ratio:.2f}x", style="yellow")
        return Text(f"✗ {ratio:.2f}x", style="red")


class extract:
    @staticmethod
    def implementation(bench: Benchmark) -> str:
        if bench.params and bench.params.id:
            return bench.params.id
        if bench.param:
            return bench.param.split("-")[0]
        if "[" in bench.name:
            return bench.name.split("[")[1].rstrip("]").split("-")[0]
        return bench.name

    @staticmethod
    def groups(benchmarks: list[Benchmark]) -> dict[str, list[Benchmark]]:
        result: dict[str, list[Benchmark]] = {}
        for b in benchmarks:
            result.setdefault(b.group, []).append(b)
        return result


class io:
    @staticmethod
    def resolve(filepath: str) -> Path:
        if (path := Path(filepath)).exists():
            return path

        if matches := glob.glob(filepath):
            resolved = Path(sorted(matches)[-1])
            console.print(f"[dim]Using: {resolved}[/dim]\n")
            return resolved

        raise FileNotFoundError(f"No file found: {filepath}")

    @staticmethod
    async def load(path: Path) -> BenchmarkData:
        async with aiofiles.open(path, "rb") as f:
            content = await f.read()
        return msgspec.json.decode(content, type=BenchmarkData, strict=False)

    @staticmethod
    async def load_many(paths: list[Path]) -> list[tuple[Path, BenchmarkData]]:
        async def load_one(p: Path) -> tuple[Path, BenchmarkData]:
            return (p, await io.load(p))

        return await asyncio.gather(*[load_one(p) for p in paths])


class report:
    @staticmethod
    def environment(data: BenchmarkData) -> None:
        lines: list[str] = []

        if data.machine_info:
            info = data.machine_info
            lines.append(f"Python: {info.python_version or 'unknown'}")
            lines.append(f"Platform: {info.platform or 'unknown'}")
            lines.append(f"CPU: {info.cpu.get('brand_raw', 'unknown')}")

        if data.commit_info and data.commit_info.id:
            lines.append(f"Commit: {data.commit_info.id[:8]}")

        header = Text()
        header.append("Benchmark Report\n", style="bold blue")
        header.append("\n".join(lines))
        console.print(Panel(header, title="Environment", border_style="blue"))
        console.print()

    @staticmethod
    def table(group_name: str, benchmarks: list[Benchmark]) -> None:
        table = Table(title=f"[bold]{group_name}[/bold]", show_header=True)
        table.add_column("Implementation", style="cyan", width=16)
        table.add_column("Min", justify="right", width=12)
        table.add_column("Mean", justify="right", width=12)
        table.add_column("Max", justify="right", width=12)
        table.add_column("StdDev", justify="right", width=12)
        table.add_column("Rounds", justify="right", width=8)
        table.add_column("OPS", justify="right", width=10)
        table.add_column("vs Best", justify="right", width=10)

        sorted_benches = sorted(benchmarks, key=lambda b: b.stats.mean)
        baseline = sorted_benches[0].stats.mean if sorted_benches else 1.0

        for bench in sorted_benches:
            stats = bench.stats
            table.add_row(
                extract.implementation(bench),
                format.time(stats.min),
                format.time(stats.mean),
                format.time(stats.max),
                format.time(stats.std),
                str(stats.rounds),
                format.operations(stats.ops),
                format.comparison(stats.mean, baseline),
            )

        console.print(table)
        console.print()

    @staticmethod
    def summary(benchmarks: list[Benchmark]) -> None:
        times: dict[str, list[float]] = {}
        for it in benchmarks:
            times.setdefault(extract.implementation(it), []).append(it.stats.mean)

        if len(times) < 2:
            return

        averages = {k: sum(v) / len(v) for k, v in times.items()}
        sorted_implementations = sorted(averages.items(), key=lambda x: x[1])
        fastest, fastest_avg = sorted_implementations[0]

        text = Text()
        text.append("Summary\n", style="bold")
        text.append(f"Total benchmarks: {len(benchmarks)}\n")
        text.append(f"Implementations: {', '.join(times.keys())}\n\n")

        for implementation, average in sorted_implementations:
            text.append(f"{implementation}: ", style="bold")
            text.append(format.time(average))
            if implementation == fastest:
                text.append(" (fastest)", style="green")
            else:
                text.append(f" ({average / fastest_avg:.2f}x slower)", style="yellow")
            text.append("\n")

        console.print(Panel(text, border_style="green"))

    @staticmethod
    def full(data: BenchmarkData) -> None:
        report.environment(data)
        for name, benches in sorted(extract.groups(data.benchmarks).items()):
            report.table(name, benches)
        report.summary(data.benchmarks)

    @staticmethod
    def comparison_table(all_data: list[tuple[Path, BenchmarkData]]) -> None:
        table = Table(title="[bold]Benchmark Comparison[/bold]")
        table.add_column("Benchmark", style="cyan")
        for path, _ in all_data:
            table.add_column(path.stem, justify="right")

        all_names: set[str] = set()
        for _, data in all_data:
            for b in data.benchmarks:
                all_names.add(f"{b.group}/{extract.implementation(b)}")

        for name in sorted(all_names):
            row: list[str] = [name]
            for _, data in all_data:
                match = next(
                    (
                        b
                        for b in data.benchmarks
                        if f"{b.group}/{extract.implementation(b)}" == name
                    ),
                    None,
                )
                row.append(format.time(match.stats.mean) if match else "-")
            table.add_row(*row)

        console.print(table)


async def show_async(filepath: str) -> None:
    try:
        path = io.resolve(filepath)
        data = await io.load(path)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except msgspec.DecodeError as e:
        console.print(f"[red]Parse error: {e}[/red]")
        raise typer.Exit(1)

    if not data.benchmarks:
        console.print("[yellow]No benchmarks found.[/yellow]")
        raise typer.Exit(0)

    report.full(data)


async def compare_async(files: list[str]) -> None:
    if len(files) < 2:
        console.print("[red]Need at least 2 files[/red]")
        raise typer.Exit(1)

    try:
        paths = [io.resolve(f) for f in files]
        all_data = await io.load_many(paths)
    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except msgspec.DecodeError as e:
        console.print(f"[red]Parse error: {e}[/red]")
        raise typer.Exit(1)

    report.comparison_table(all_data)


@app.command()
def show(
    filepath: str = typer.Argument(
        "benchmark.json", help="Benchmark JSON path (supports globs)"
    ),
) -> None:
    """Generate a pretty benchmark report."""
    asyncio.run(show_async(filepath))


@app.command()
def compare(
    files: list[str] = typer.Argument(..., help="Two or more JSON files"),
) -> None:
    """Compare benchmarks across multiple runs."""
    asyncio.run(compare_async(files))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
