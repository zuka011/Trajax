import asyncio
from typing import Final, Sequence
from pathlib import Path

from .parse import NotebookParser

from aiopath import AsyncPath


class defaults:
    examples_directory: Final = Path("docs/examples")
    output_directory: Final = Path("notebooks")
    install_command: Final = "!pip install -q trajax trajax-visualizer"


async def generate(
    *,
    examples_dir: Path = defaults.examples_directory,
    output_dir: Path = defaults.output_directory,
    install_command: str = defaults.install_command,
) -> Sequence[Path]:
    """Generate notebooks for every example and return the output paths."""
    await AsyncPath(output_dir).mkdir(parents=True, exist_ok=True)

    example_files = sorted(examples_dir.glob("[0-9]*.py"))
    parser = NotebookParser(install_command=install_command)

    async def process(source_path: Path) -> Path:
        source = await AsyncPath(source_path).read_text(encoding="utf-8")
        notebook = parser.parse(source)
        output_path = output_dir / (source_path.stem + ".ipynb")
        await AsyncPath(output_path).write_text(notebook.json(), encoding="utf-8")
        return output_path

    return await asyncio.gather(*[process(path) for path in example_files])
