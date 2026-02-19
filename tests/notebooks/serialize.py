from typing import Final, Protocol, Sequence, Any
from dataclasses import dataclass, field

from .cells import Cell, CellType, cell

import msgspec

NOTEBOOK_VERSION: Final = 4
KERNEL_SPEC: Final = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
LANGUAGE_INFO: Final = {
    "name": "python",
    "version": "3.13.0",
    "mimetype": "text/x-python",
    "file_extension": ".py",
}


class Notebook(Protocol):
    @property
    def title(self) -> str:
        """Return the title of this notebook."""
        ...

    @property
    def description(self) -> str:
        """Return the description of this notebook."""
        ...

    @property
    def cells(self) -> Sequence[Cell]:
        """Return the cells of this notebook in order."""
        ...

    @property
    def header(self) -> Cell:
        """Return the combined title and description as a markdown header."""
        ...


class SerializeNotebookMixin:
    @property
    def header(self: Notebook) -> Cell:
        """Return the combined title and description as a markdown header."""
        header = f"# {self.title}"

        if self.description != "":
            header += f"\n\n{self.description}"

        return cell.markdown(header)

    def json(self: Notebook) -> str:
        """Serialize this notebook to a JSON string in the Jupyter format."""
        cells = [as_jupyter_cell(self.header)] + [
            as_jupyter_cell(cell) for cell in self.cells
        ]

        notebook = JupyterNotebook(
            nbformat=NOTEBOOK_VERSION,
            nbformat_minor=5,
            metadata=NotebookMetadata(
                kernelspec=KERNEL_SPEC, language_info=LANGUAGE_INFO
            ),
            cells=cells,
        )

        encoded = msgspec.json.encode(notebook)
        formatted = msgspec.json.format(encoded, indent=1)

        return formatted.decode("utf-8") + "\n"


@dataclass(frozen=True)
class JupyterMarkdownCell:
    cell_type: str
    metadata: dict[str, Any]
    source: list[str]


@dataclass(frozen=True)
class JupyterCodeCell:
    cell_type: str
    metadata: dict[str, Any]
    source: list[str]
    outputs: list[Any] = field(default_factory=list)
    execution_count: int | None = None


type JupyterCell = JupyterMarkdownCell | JupyterCodeCell


@dataclass(frozen=True)
class NotebookMetadata:
    kernelspec: dict[str, str]
    language_info: dict[str, str]


@dataclass(frozen=True)
class JupyterNotebook:
    nbformat: int
    nbformat_minor: int
    metadata: NotebookMetadata
    cells: list[JupyterCell]


def as_source_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []


def as_jupyter_cell(cell: Cell) -> JupyterCell:
    source_lines = as_source_lines(cell.content)

    match cell.type:
        case CellType.MARKDOWN:
            return JupyterMarkdownCell(
                cell_type="markdown", metadata={}, source=source_lines
            )
        case CellType.CODE:
            return JupyterCodeCell(
                cell_type="code",
                metadata={},
                source=source_lines,
                outputs=[],
                execution_count=None,
            )
