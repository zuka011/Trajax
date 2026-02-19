import asyncio
from typing import Annotated
from pathlib import Path

from .generate import generate, defaults

import typer


app = typer.Typer(
    help="Generate Jupyter notebooks from documentation example scripts.",
    no_args_is_help=False,
)


@app.command()
def run(
    examples_dir: Annotated[
        Path,
        typer.Option("--examples", "-e", help="Directory containing example .py files"),
    ] = defaults.examples_directory,
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory for generated .ipynb files"),
    ] = defaults.output_directory,
    install_command: Annotated[
        str,
        typer.Option("--install", "-i", help="pip install command for notebooks"),
    ] = defaults.install_command,
) -> None:
    generated = asyncio.run(
        generate(
            examples_dir=examples_dir,
            output_dir=output_dir,
            install_command=install_command,
        )
    )

    if not generated:
        typer.echo("No examples found.", err=True)
        raise typer.Exit(1)

    for path in generated:
        typer.echo(f"  {path}")

    typer.echo(f"\nGenerated {len(generated)} notebook(s).")


def main() -> None:
    app()
