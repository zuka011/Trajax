from pathlib import Path

from . import formatting, parsing
from .models import Benchmark, BenchmarkData

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def environment(data: BenchmarkData, console: Console) -> None:
    lines: list[str] = []

    if data.machine_info:
        info = data.machine_info
        gpus = info.gpu.get("devices", [])
        lines.append(f"Python: {info.python_version or 'unknown'}")
        lines.append(f"Platform: {info.platform or 'unknown'}")
        lines.append(f"CPU: {info.cpu.get('brand_raw', 'unknown')}")
        lines.append(
            f"GPU: {', '.join(it.get('name', 'unknown') for it in gpus)} - {info.gpu.get('count', 0)} device(s)"
            if len(gpus) > 0
            else "GPU: none"
        )

    if data.commit_info and data.commit_info.id:
        lines.append(f"Commit: {data.commit_info.id[:8]}")

    header = Text()
    header.append("Benchmark Report\n", style="bold blue")
    header.append("\n".join(lines))
    console.print(Panel(header, title="Environment", border_style="blue"))
    console.print()


def table(group_name: str, benchmarks: list[Benchmark], console: Console) -> None:
    tbl = Table(title=f"[bold]{group_name}[/bold]", show_header=True)
    tbl.add_column("Implementation", style="cyan", width=16)
    tbl.add_column("Min", justify="right", width=12)
    tbl.add_column("Mean", justify="right", width=12)
    tbl.add_column("Max", justify="right", width=12)
    tbl.add_column("StdDev", justify="right", width=12)
    tbl.add_column("Rounds", justify="right", width=8)
    tbl.add_column("OPS", justify="right", width=10)
    tbl.add_column("vs Best", justify="right", width=10)

    sorted_benchmarks = sorted(benchmarks, key=lambda b: b.stats.mean)
    baseline = sorted_benchmarks[0].stats.mean if sorted_benchmarks else 1.0

    for benchmark in sorted_benchmarks:
        stats = benchmark.stats
        tbl.add_row(
            parsing.implementation(benchmark),
            formatting.time(stats.min),
            formatting.time(stats.mean),
            formatting.time(stats.max),
            formatting.time(stats.stddev),
            str(stats.rounds),
            formatting.operations(stats.ops),
            formatting.comparison(stats.mean, baseline),
        )

    console.print(tbl)
    console.print()


def summary(benchmarks: list[Benchmark], console: Console) -> None:
    times: dict[str, list[float]] = {}
    for benchmark in benchmarks:
        times.setdefault(parsing.implementation(benchmark), []).append(
            benchmark.stats.mean
        )

    if len(times) < 2:
        return

    averages = {k: sum(v) / len(v) for k, v in times.items()}
    sorted_implementations = sorted(averages.items(), key=lambda x: x[1])
    fastest, fastest_avg = sorted_implementations[0]

    text = Text()
    text.append("Summary\n", style="bold")
    text.append(f"Total benchmarks: {len(benchmarks)}\n")
    text.append(f"Implementations: {', '.join(times.keys())}\n\n")

    for impl, avg in sorted_implementations:
        text.append(f"{impl}: ", style="bold")
        text.append(formatting.time(avg))
        if impl == fastest:
            text.append(" (fastest)", style="green")
        else:
            text.append(f" ({avg / fastest_avg:.2f}x slower)", style="yellow")
        text.append("\n")

    console.print(Panel(text, border_style="green"))


def full(data: BenchmarkData, console: Console) -> None:
    environment(data, console)
    for name, benchmarks in sorted(parsing.groups(data.benchmarks).items()):
        table(name, benchmarks, console)
    summary(data.benchmarks, console)


def comparison_table(
    all_data: list[tuple[Path, BenchmarkData]], console: Console
) -> None:
    tbl = Table(title="[bold]Benchmark Comparison[/bold]")
    tbl.add_column("Benchmark", style="cyan")
    for path, _ in all_data:
        tbl.add_column(path.stem, justify="right")

    all_names: set[str] = set()
    for _, data in all_data:
        for benchmark in data.benchmarks:
            all_names.add(
                f"{parsing.group(benchmark)}/{parsing.implementation(benchmark)}"
            )

    for name in sorted(all_names):
        row: list[str] = [name]
        for _, data in all_data:
            match = next(
                (
                    b
                    for b in data.benchmarks
                    if f"{parsing.group(b)}/{parsing.implementation(b)}" == name
                ),
                None,
            )
            row.append(formatting.time(match.stats.mean) if match else "-")
        tbl.add_row(*row)

    console.print(tbl)
