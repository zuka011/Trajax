import pytest

pytest.register_assert_rewrite("tests.dsl")

from typing import AsyncGenerator
from pathlib import Path

from trajax_visualizer import configure as configure_visualizer

from tests.visualize import add_visualizer_option, visualization as visualization
from tests.tasks import BackgroundTasks
from pytest_asyncio import fixture as async_fixture


def pytest_addoption(parser: pytest.Parser) -> None:
    add_visualizer_option(parser)


def pytest_configure(config: pytest.Config) -> None:
    configure_visualizer(output_directory=Path(__file__).parent / "visualizations")


@async_fixture(scope="session")
async def background_tasks() -> AsyncGenerator[BackgroundTasks, None]:
    async with BackgroundTasks() as tasks:
        yield tasks
