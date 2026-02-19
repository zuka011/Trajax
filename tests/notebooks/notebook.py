from typing import Sequence
from dataclasses import dataclass

from .cells import Cell
from .serialize import SerializeNotebookMixin


@dataclass(frozen=True)
class Notebook(SerializeNotebookMixin):
    title: str
    description: str
    cells: Sequence[Cell]
