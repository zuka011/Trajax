from typing import Protocol, AsyncGenerator, Final
from dataclasses import dataclass

from pytest import Parser, FixtureRequest
from pytest_asyncio import fixture as async_fixture


class Visualizer[T](Protocol):
    async def __call__(self, data: T) -> None:
        """Visualizes the given data."""
        ...

    async def can_visualize(self, data: object) -> bool:
        """Returns whether this visualizer can visualize the given data."""
        ...


@dataclass
class VisualizationData[T]:
    _data: T | None

    @staticmethod
    def empty[T_ = object]() -> "VisualizationData[T_]":
        return VisualizationData(None)

    def data_is(self, data: T) -> None:
        self._data = data

    @property
    def data(self) -> T | None:
        return self._data


class options:
    visualize: Final = "--visualize"


class markers:
    visualize: Final = "visualize"


def add_visualizer_option(parser: Parser) -> None:
    parser.addoption(
        options.visualize,
        action="store_true",
        default=False,
        help="Run visualization after tests marked with @pytest.mark.visualize",
    )


async def visualizer_from[T](
    request: FixtureRequest, capture: VisualizationData[T]
) -> Visualizer[T] | None:
    if (
        not request.config.getoption(options.visualize)
        or (marker := request.node.get_closest_marker(markers.visualize)) is None
    ):
        return

    assert len(marker.args) == 1, (
        f"{markers.visualize} requires a single visualizer argument, got {len(marker.args)}. "
        "Example: @pytest.mark.visualize.with_args(my_visualizer)"
    )

    visualizer: Visualizer = marker.args[0]

    assert await visualizer.can_visualize(capture.data), (
        f"Visualizer {visualizer} cannot visualize data of type "
        f"{type(capture.data).__name__}"
    )

    return visualizer


@async_fixture
async def visualization[T = object](
    request: FixtureRequest,
) -> AsyncGenerator[VisualizationData[T] | None, None]:
    """Captures visualization data and triggers the visualizer after the test."""
    capture: VisualizationData[T] = VisualizationData.empty()
    yield capture

    if (
        capture.data is None
        or (visualizer := await visualizer_from(request, capture)) is None
    ):
        return

    await visualizer(capture.data)
