import pytest

pytest.register_assert_rewrite("tests.dsl")

from tests.visualize import add_visualizer_option, visualization as visualization


def pytest_addoption(parser: pytest.Parser) -> None:
    add_visualizer_option(parser)
