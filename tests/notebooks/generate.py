import asyncio
from typing import Final, Sequence
from pathlib import Path

from .cells import Cell, cell
from .parse import NotebookParser

from aiopath import AsyncPath


DISPLAY_CODE: Final = """\
from IPython.display import IFrame, display as show_inline
show_inline(IFrame("mpcc-simulation/visualization.html", width="100%", height=600))
""".strip()


class defaults:
    examples_directory: Final = Path("docs/examples")
    output_directory: Final = Path("notebooks")
    install_command: Final = "!pip install -q faran faran-visualizer"
    visualization_cells: Final[Sequence[Cell]] = (cell.code(DISPLAY_CODE),)


async def generate(
    *,
    examples_dir: Path = defaults.examples_directory,
    output_dir: Path = defaults.output_directory,
    install_command: str = defaults.install_command,
    trailing_cells: Sequence[Cell] = defaults.visualization_cells,
) -> Sequence[Path]:
    """Generate notebooks for every example and return the output paths."""
    await AsyncPath(output_dir).mkdir(parents=True, exist_ok=True)

    example_files = sorted(examples_dir.glob("[0-9]*.py"))
    parser = NotebookParser(
        install_command=install_command, trailing_cells=trailing_cells
    )

    async def process(source_path: Path) -> Path:
        source = await AsyncPath(source_path).read_text(encoding="utf-8")
        notebook = parser.parse(source)
        output_path = output_dir / (source_path.stem + ".ipynb")
        await AsyncPath(output_path).write_text(notebook.json(), encoding="utf-8")
        return output_path

    return await asyncio.gather(*[process(path) for path in example_files])
