import asyncio
from typing import Any
from pathlib import Path
from io import StringIO

from . import io, parsing, rendering

from rich.console import Console

import msgspec
import typer

app = typer.Typer(
    help="Generate pretty benchmark reports and comparisons from pytest-benchmark JSON.",
    no_args_is_help=True,
)
console = Console()


async def show_async(
    filepath: str,
    output_html: Path | None = None,
    output_json: Path | None = None,
) -> None:
    """Async implementation of show command."""
    try:
        path = io.resolve(filepath, console)
        data = await io.load(path)
    except FileNotFoundError as error:
        console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(1)
    except msgspec.DecodeError as error:
        console.print(f"[red]Parse error: {error}[/red]")
        raise typer.Exit(1)

    if not data.benchmarks:
        console.print("[yellow]No benchmarks found.[/yellow]")
        raise typer.Exit(0)

    rendering.full(data, console)

    save_tasks: list[Any] = []

    if output_html:
        html_console = Console(
            record=True, force_terminal=True, width=120, file=StringIO()
        )
        rendering.full(data, html_console)
        save_tasks.append(io.save_html(output_html, html_console, console))

    if output_json:
        export = parsing.export_data(data)
        save_tasks.append(io.save_json(output_json, export, console))

    if save_tasks:
        await asyncio.gather(*save_tasks)


async def compare_async(files: list[str]) -> None:
    if len(files) < 2:
        console.print("[red]Need at least 2 files[/red]")
        raise typer.Exit(1)

    try:
        paths = [io.resolve(file, console) for file in files]
        all_data = await io.load_many(paths)
    except FileNotFoundError as error:
        console.print(f"[red]Error: {error}[/red]")
        raise typer.Exit(1)
    except msgspec.DecodeError as error:
        console.print(f"[red]Parse error: {error}[/red]")
        raise typer.Exit(1)

    rendering.comparison_table(all_data, console)


@app.command()
def show(
    filepath: str = typer.Argument(
        "benchmark.json", help="Benchmark JSON path (supports globs)"
    ),
    output_html: Path | None = typer.Option(
        "benchmark_report.html",
        "--html",
        "-h",
        help="Export report to HTML file",
    ),
    output_json: Path | None = typer.Option(
        "benchmark_report.json",
        "--json",
        "-j",
        help="Export parsed data to JSON file",
    ),
) -> None:
    """Generate a pretty benchmark report."""
    asyncio.run(show_async(filepath, output_html, output_json))


@app.command()
def compare(
    files: list[str] = typer.Argument(..., help="Two or more JSON files"),
) -> None:
    """Compare benchmarks across multiple runs."""
    asyncio.run(compare_async(files))


def main() -> None:
    """Entry point for CLI."""
    app()
