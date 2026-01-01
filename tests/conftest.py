import pytest

pytest.register_assert_rewrite("tests.dsl")

from typing import AsyncGenerator

from tests.visualize import add_visualizer_option, visualization as visualization
from tests.tasks import BackgroundTasks
from pytest_asyncio import fixture as async_fixture


def pytest_addoption(parser: pytest.Parser) -> None:
    add_visualizer_option(parser)


@async_fixture(scope="session")
async def background_tasks() -> AsyncGenerator[BackgroundTasks, None]:
    async with BackgroundTasks() as tasks:
        yield tasks
