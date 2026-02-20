import re
import textwrap
from typing import Final, Sequence, NamedTuple
from dataclasses import dataclass, field

from .cells import Cell, cell
from .notebook import Notebook


SECTION_PATTERN: Final = re.compile(r"^#\s*──\s*(.+?)\s*─+\s*#\s*$")
SNIPPET_MARKER: Final = re.compile(r"^\s*#\s*--8<--\s*\[(?:start|end):\w+\]\s*$")
MAIN_GUARD: Final = re.compile(r'^if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*$')
TEST_CONSTANT: Final = re.compile(r"^(?:SEED|MAX_\w+|HAS_\w+|GOAL_FRACTION)\s*=\s*")
ASYNCIO_RUN: Final = re.compile(r"^(\s*)asyncio\.run\((.+)\)\s*$")


class DocstringParseResult(NamedTuple):
    docstring: str
    remaining_lines: list[str]


class HeaderParseResult(NamedTuple):
    title: str
    description: str


class MainBlockParseResult(NamedTuple):
    before_main: list[str]
    main_body: str


class Section(NamedTuple):
    header: str | None
    lines: list[str]


class transform:
    @staticmethod
    def strip_snippet_markers(lines: list[str]) -> list[str]:
        return [line for line in lines if not SNIPPET_MARKER.match(line)]

    @staticmethod
    def strip_test_constants(lines: list[str]) -> list[str]:
        return [line for line in lines if not TEST_CONSTANT.match(line)]

    @staticmethod
    def replace_asyncio_run_with_await(lines: list[str]) -> list[str]:
        return [ASYNCIO_RUN.sub(r"\1await \2", line) for line in lines]

    @staticmethod
    def strip_blanks(lines: list[str]) -> list[str]:
        start = 0
        end = len(lines)

        while start < end and not lines[start].strip():
            start += 1

        while end > start and not lines[end - 1].strip():
            end -= 1

        return lines[start:end]

    @staticmethod
    def clean(lines: list[str]) -> list[str]:
        result = transform.strip_snippet_markers(lines)
        result = transform.strip_test_constants(result)
        result = transform.strip_blanks(result)
        return result


class parse:
    @staticmethod
    def docstring(lines: list[str]) -> DocstringParseResult:
        def is_empty_docstring() -> bool:
            return len(lines) == 0 or not lines[0].strip().startswith('"""')

        def is_single_line_docstring() -> bool:
            return lines[0].count('"""') >= 2

        def extract_empty_docstring() -> DocstringParseResult:
            return DocstringParseResult(docstring="", remaining_lines=lines[1:])

        def extract_single_line_docstring() -> DocstringParseResult:
            docstring = lines[0].strip().strip('"""').strip()
            return DocstringParseResult(docstring=docstring, remaining_lines=lines[1:])

        def extract_multi_line_docstring() -> DocstringParseResult:
            docstring_lines: list[str] = []
            index = 0

            for index, line in enumerate(lines):
                docstring_lines.append(line)
                if index > 0 and '"""' in line:
                    break

            docstring = "\n".join(docstring_lines).strip().strip('"""').strip()
            return DocstringParseResult(
                docstring=docstring, remaining_lines=lines[index + 1 :]
            )

        if is_empty_docstring():
            return DocstringParseResult(docstring="", remaining_lines=lines)

        if is_single_line_docstring():
            return extract_single_line_docstring()

        return extract_multi_line_docstring()

    @staticmethod
    def header(lines: list[str]) -> HeaderParseResult:
        if not lines:
            return HeaderParseResult(title="Untitled", description="")

        title = lines[0].strip()
        description = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        return HeaderParseResult(title=title, description=description)

    @staticmethod
    def main_block(lines: list[str]) -> MainBlockParseResult:
        if (
            main_start := next(
                (i for i, line in enumerate(lines) if MAIN_GUARD.match(line)), None
            )
        ) is None:
            return MainBlockParseResult(before_main=lines, main_body="")

        body = [line.removeprefix("    ") for line in lines[main_start + 1 :]]

        return MainBlockParseResult(
            before_main=lines[:main_start], main_body="\n".join(transform.clean(body))
        )

    @staticmethod
    def sections(lines: list[str]) -> list[Section]:
        sections: list[Section] = []
        current_header: str | None = None
        current_lines: list[str] = []

        def flush() -> None:
            if len(current_lines) > 0 or current_header is not None:
                sections.append(Section(header=current_header, lines=current_lines))

        for line in lines:
            if match := SECTION_PATTERN.match(line):
                flush()
                current_header = match.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)

        flush()
        return sections


@dataclass(kw_only=True, frozen=True)
class NotebookParser:
    install_command: str
    trailing_cells: Sequence[Cell] = field(default_factory=tuple)

    def parse(self, source: str) -> Notebook:
        """Build a notebook from a Python example source file."""
        dedented = textwrap.dedent(source)
        docstring, remaining = parse.docstring(transform.clean(dedented.splitlines()))
        title, description = parse.header(docstring.splitlines())

        cells: list[Cell] = [cell.code(self.install_command)]

        def combine(lines: list[str]) -> str:
            return "\n".join(transform.clean(lines))

        for section_name, section_lines in parse.sections(remaining):
            body, main_body = parse.main_block(section_lines)

            if code := combine(body):
                if section_name:
                    cells.append(cell.markdown(f"## {section_name}"))

                cells.append(cell.code(code))

            if main_body:
                cells.append(cell.markdown("## Run"))
                cells.append(
                    cell.code(
                        combine(
                            transform.replace_asyncio_run_with_await(
                                main_body.splitlines()
                            )
                        )
                    )
                )

        cells.extend(self.trailing_cells)

        return Notebook(title=title, description=description, cells=cells)
