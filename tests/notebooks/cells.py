from typing import Protocol, Final
from dataclasses import dataclass, field
from enum import StrEnum


class CellType(StrEnum):
    MARKDOWN = "markdown"
    CODE = "code"


class Cell(Protocol):
    @property
    def content(self) -> str:
        """Return the content of this cell as a string."""
        ...

    @property
    def type(self) -> CellType:
        """Return the type of this cell."""
        ...


@dataclass(frozen=True)
class MarkdownCell:
    content: str
    type: CellType = field(default=CellType.MARKDOWN, init=False)


@dataclass(frozen=True)
class CodeCell:
    content: str
    type: CellType = field(default=CellType.CODE, init=False)


class cell:
    markdown: Final = MarkdownCell
    code: Final = CodeCell
