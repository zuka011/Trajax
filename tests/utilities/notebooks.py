import asyncio

import pytest

from tests.notebooks import generate
from tests.utilities.root import project_root


def add_notebook_option(parser: pytest.Parser) -> None:
    parser.addoption(
        "--notebooks",
        action="store_true",
        default=False,
        help="Generate Jupyter notebooks from documentation examples after tests pass",
    )


def is_notebook_generation_enabled(session: pytest.Session) -> bool:
    return session.config.getoption("--notebooks")


def generate_notebooks() -> str:
    """Generate notebooks from documentation examples and return a summary."""

    root = project_root()
    generated = asyncio.run(
        generate(examples_dir=root / "docs" / "examples", output_dir=root / "notebooks")
    )

    if not generated:
        return "No documentation examples found for notebook generation."

    paths = "\n".join(f"  {path}" for path in generated)
    return f"\n{paths}\n\nGenerated {len(generated)} notebook(s)."
