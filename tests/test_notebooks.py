import json
from pathlib import Path

from tests.notebooks import NotebookParser, Notebook, CellType, generate
from pytest import mark


def notebook_from_example() -> Notebook:
    return NotebookParser(install_command="!pip install foo").parse(
        '''\
    """Title of the example.

    A longer description that spans
    multiple lines.
    """

    from foo import bar

    # ── Constants ──────────────────────────────────────────────────────────────── #

    X = 1
    Y = 2

    # ── Setup ──────────────────────────────────────────────────────────────────── #

    # --8<-- [start:setup]
    def create():
    return bar(X, Y)
    # --8<-- [end:setup]

    SEED = "test-seed"
    MAX_ERROR = 5.0

    if __name__ == "__main__":
    result = create()
    print(result)
    ''',
    )


def json_from_example() -> dict:
    return json.loads(notebook_from_example().json())


def test_that_notebook_extracts_title_from_docstring() -> None:
    assert notebook_from_example().title == "Title of the example."


def test_that_notebook_extracts_description_from_docstring() -> None:
    assert (
        notebook_from_example().description
        == "A longer description that spans\nmultiple lines."
    )


def test_that_notebook_defaults_to_untitled_when_there_is_no_docstring() -> None:
    notebook = NotebookParser(install_command="!pip install foo").parse("import foo\n")

    assert notebook.title == "Untitled"
    assert notebook.description == ""


def test_that_notebook_starts_with_install_cell() -> None:
    first = notebook_from_example().cells[0]

    assert first.content == "!pip install foo"


def test_that_notebook_includes_preamble_code() -> None:
    code_sources = [
        cell.content
        for cell in notebook_from_example().cells
        if cell.type == CellType.CODE
    ]

    assert any("from foo import bar" in source for source in code_sources)


def test_that_notebook_includes_section_headers() -> None:
    headers = [
        cell.content
        for cell in notebook_from_example().cells
        if cell.type == CellType.MARKDOWN
    ]

    assert "## Constants" in headers
    assert "## Setup" in headers


def test_that_notebook_does_not_include_snippet_markers() -> None:
    assert all("--8<--" not in cell.content for cell in notebook_from_example().cells)


def test_that_notebook_does_not_include_test_constants() -> None:
    code_sources = [
        cell.content
        for cell in notebook_from_example().cells
        if cell.type == CellType.CODE
    ]

    assert not any("SEED =" in source for source in code_sources)
    assert not any("MAX_ERROR =" in source for source in code_sources)


def test_that_notebook_includes_run_section_from_main_guard() -> None:
    cells = notebook_from_example().cells
    run_indices = [
        i
        for i, cell in enumerate(cells)
        if cell.type == CellType.MARKDOWN and cell.content == "## Run"
    ]

    assert len(run_indices) == 1
    run_cell = cells[run_indices[0] + 1]
    assert run_cell.type == CellType.CODE
    assert "create()" in run_cell.content
    assert "print(" in run_cell.content


def test_that_notebook_omits_main_guard_syntax() -> None:
    assert all("__name__" not in cell.content for cell in notebook_from_example().cells)


def test_that_notebook_json_has_valid_format() -> None:
    result = json_from_example()

    assert result["nbformat"] == 4
    assert result["nbformat_minor"] == 5
    assert "kernelspec" in result["metadata"]
    assert "language_info" in result["metadata"]


def test_that_notebook_json_starts_with_title_cell() -> None:
    first = json_from_example()["cells"][0]
    combined = "".join(first["source"])

    assert first["cell_type"] == "markdown"
    assert "# Title of the example" in combined
    assert "A longer description" in combined


def test_that_notebook_json_includes_source_formatted_as_line_list() -> None:
    for cell in json_from_example()["cells"]:
        assert isinstance(cell["source"], list)
        assert all(isinstance(line, str) for line in cell["source"])


def test_that_notebook_json_code_cells_have_empty_outputs() -> None:
    code_cells = [
        cell for cell in json_from_example()["cells"] if cell["cell_type"] == "code"
    ]

    assert all(cell["outputs"] == [] for cell in code_cells)
    assert all(cell["execution_count"] is None for cell in code_cells)


@mark.asyncio
async def test_that_generate_writes_notebooks_for_example_files(
    tmp_path: Path,
) -> None:
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()

    (examples_dir / "01_demo.py").write_text(
        '"""Demo.\n\nA demo example.\n"""\n\nimport math\n\nX = math.pi\n',
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"

    generated = await generate(
        examples_dir=examples_dir,
        output_dir=output_dir,
        install_command="!pip install demo",
    )

    assert len(generated) == 1
    assert generated[0].name == "01_demo.ipynb"
    assert generated[0].exists()


@mark.asyncio
async def test_that_generate_produces_valid_notebook_json(tmp_path: Path) -> None:
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()

    (examples_dir / "01_demo.py").write_text(
        '"""Demo.\n\nA demo example.\n"""\n\nimport math\n\nX = math.pi\n',
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"

    generated = await generate(
        examples_dir=examples_dir,
        output_dir=output_dir,
        install_command="!pip install demo",
    )

    content = json.loads(generated[0].read_text(encoding="utf-8"))

    assert content["nbformat"] == 4
    assert "cells" in content
    assert any(
        "math" in "".join(cell["source"])
        for cell in content["cells"]
        if cell["cell_type"] == "code"
    )


@mark.asyncio
async def test_that_generate_processes_multiple_examples_concurrently(
    tmp_path: Path,
) -> None:
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()

    for i in range(1, 4):
        (examples_dir / f"0{i}_example.py").write_text(
            f'"""Example {i}."""\n\nX = {i}\n', encoding="utf-8"
        )

    output_dir = tmp_path / "output"

    generated = await generate(
        examples_dir=examples_dir,
        output_dir=output_dir,
        install_command="!pip install demo",
    )

    assert len(generated) == 3
    assert all(path.exists() for path in generated)
