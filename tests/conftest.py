import pytest

pytest.register_assert_rewrite("tests.dsl")
pytest.register_assert_rewrite("docs.examples")

from typing import AsyncGenerator

from trajax_visualizer import configure as configure_visualizer

from tests.utilities import (
    project_root,
    add_visualizer_option,
    add_compilation_tracker_option,
    is_compilation_tracker_enabled,
    compilation_tracker,
    add_notebook_option,
    is_notebook_generation_enabled,
    generate_notebooks,
    visualization as visualization,
)
from tests.tasks import BackgroundTasks
from pytest_asyncio import fixture as async_fixture


def pytest_addoption(parser: pytest.Parser) -> None:
    add_visualizer_option(parser)
    add_compilation_tracker_option(parser)
    add_notebook_option(parser)


def pytest_configure(config: pytest.Config) -> None:
    configure_visualizer(output_directory=project_root() / "docs" / "visualizations")


def pytest_sessionstart(session: pytest.Session) -> None:
    if is_compilation_tracker_enabled(session):
        compilation_tracker.start()


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if is_compilation_tracker_enabled(session):
        print(f"\n\n{compilation_tracker.stop()}")

    if is_notebook_generation_enabled(session) and exitstatus == 0:
        print(f"\n{generate_notebooks()}")


@async_fixture(scope="session")
async def background_tasks() -> AsyncGenerator[BackgroundTasks, None]:
    async with BackgroundTasks() as tasks:
        yield tasks
