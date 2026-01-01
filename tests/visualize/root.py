from typing import Final
from pathlib import Path


ROOT_MARKER: Final = "pyproject.toml"


def find_root() -> Path:
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / ROOT_MARKER).exists():
            return parent

    assert False, "Project root could not be found."
