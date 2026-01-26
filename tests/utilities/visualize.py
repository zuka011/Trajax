from typing import (
    Protocol,
    AsyncGenerator,
    Final,
    NamedTuple,
    Self,
    runtime_checkable,
    cast,
)
from dataclasses import dataclass

from tests.tasks import BackgroundTasks

from pytest import Parser, FixtureRequest, param, mark
from pytest_asyncio import fixture as async_fixture


@runtime_checkable
class Visualizer[T](Protocol):
    async def __call__(self, data: T, *, key: str) -> None:
        """Visualizes the given data. The key can be used to distinguish between
        different visualization contexts."""
        ...

    async def can_visualize(self, data: object) -> bool:
        """Returns whether this visualizer can visualize the given data."""
        ...


@runtime_checkable
class KeyProvider(Protocol):
    def __call__(self, seed: str, /) -> str:
        """Returns a key to distinguish between different visualization contexts."""
        ...


@dataclass
class VisualizationData[T]:
    _data: T | None
    _seed: str | None

    @staticmethod
    def empty[T_ = object]() -> "VisualizationData[T_]":
        return VisualizationData(None, None)

    def data_is(self, data: T) -> Self:
        self._data = data
        return self

    def seed_is(self, seed: str) -> Self:
        self._seed = seed
        return self

    @property
    def data(self) -> T | None:
        return self._data

    @property
    def seed(self) -> str | None:
        return self._seed


class VisualizerArguments[T](NamedTuple):
    visualizer: Visualizer[T]
    key: str


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


async def visualizer_arguments_from[T](
    request: FixtureRequest, capture: VisualizationData[T]
) -> VisualizerArguments[T] | None:
    if (
        not request.config.getoption(options.visualize)
        or (marker := request.node.get_closest_marker(markers.visualize)) is None
    ):
        return

    match marker.args:
        case (Visualizer() as visualizer, str() as key):
            pass
        case (Visualizer() as visualizer, KeyProvider() as key_provider):
            assert (
                capture.seed is not None
            ), """A key provider was used for visualization, but no seed was
                provided during the test.
                
                Example: @pytest.mark.visualize.with_args(my_visualizer, lambda seed: f"prefix-{seed}")

                Then inside the test:
                    capture.seed_is("my-test-seed")
                """
            key = key_provider(capture.seed)
        case _:
            assert False, f"""{markers.visualize} expects the following arguments: 
                    - a Visualizer instance. 
                    - a key to distinguish between different visualization contexts.
                
                Example: @pytest.mark.visualize.with_args(my_visualizer, "my-test-key")
                """

    assert await visualizer.can_visualize(capture.data), (
        f"Visualizer {visualizer} cannot visualize data of type "
        f"{type(capture.data).__name__}"
    )

    return VisualizerArguments(visualizer=visualizer, key=key)


@async_fixture
async def visualization[T = object](
    request: FixtureRequest,
    background_tasks: BackgroundTasks,
) -> AsyncGenerator[VisualizationData[T] | None, None]:
    """Captures visualization data and triggers the visualizer after the test."""
    capture: VisualizationData[T] = VisualizationData.empty()
    yield capture

    if (
        capture.data is None
        or (args := await visualizer_arguments_from(request, capture)) is None
    ):
        return

    background_tasks.schedule(args.visualizer(capture.data, key=args.key))


def doc_example[*Args](*case: *Args) -> tuple[*Args]:
    """Marks a parameterized test case as a documentation example."""
    return cast(tuple[*Args], param(*case, marks=mark.docs))
