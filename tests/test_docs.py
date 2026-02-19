from typing import Sequence, Final, Callable, Protocol
from dataclasses import dataclass
from types import ModuleType
from importlib import import_module
from pathlib import Path
from functools import lru_cache

from trajax import Trajectory
from trajax_visualizer import MpccSimulationResult, visualizer

from tests.utilities import VisualizationData, doc_example
from pytest import mark

EXAMPLES_DIRECTORY: Final = Path("docs/examples")


class ExampleResult(Protocol):
    reached_goal: bool
    progress: float
    visualization: MpccSimulationResult


@dataclass(frozen=True)
class ExampleModule[*ComponentsT]:
    seed: int
    reference: Trajectory
    goal_fraction: float
    max_contouring_error: float
    max_lag_error: float
    create: Callable[[], *ComponentsT]
    run: Callable[[*ComponentsT], ExampleResult]


@lru_cache
def discover() -> Sequence[str]:
    """Return module paths for all numbered example scripts."""
    return [
        f"docs.examples.{path.stem}"
        for path in sorted(EXAMPLES_DIRECTORY.glob("[0-9]*.py"))
    ]


class extract:
    @staticmethod
    def module_id(module_path: str) -> str:
        return module_path.rsplit(".", 1)[-1]

    @staticmethod
    def component_creator(module: ModuleType) -> Callable[[], Sequence]:
        assert hasattr(module, "create"), (
            f"Module {module.__name__} must define a create() function "
            f"that returns the components for the test"
        )

        return getattr(module, "create")

    @staticmethod
    def simulation_runner(
        module: ModuleType,
    ) -> Callable[[Sequence], MpccSimulationResult]:
        assert hasattr(module, "run"), (
            f"Module {module.__name__} must define a run(*components) "
            f"function that executes the simulation and returns an MpccSimulationResult"
        )

        return getattr(module, "run")

    @staticmethod
    def seed(module: ModuleType) -> int:
        assert hasattr(module, "SEED"), (
            f"Module {module.__name__} must define a SEED variable"
        )

        return getattr(module, "SEED")

    @staticmethod
    def reference(module: ModuleType) -> Trajectory:
        assert hasattr(module, "REFERENCE"), (
            f"Module {module.__name__} must define a REFERENCE variable "
            f"of type Trajectory"
        )

        return getattr(module, "REFERENCE")

    @staticmethod
    def goal_fraction(module: ModuleType) -> float:
        return getattr(module, "GOAL_FRACTION", 0.9)

    @staticmethod
    def max_contouring_error(module: ModuleType) -> float:
        assert hasattr(module, "MAX_CONTOURING_ERROR"), (
            f"Module {module.__name__} must define a MAX_CONTOURING_ERROR "
            f"variable specifying the maximum allowed contouring error in meters"
        )

        return getattr(module, "MAX_CONTOURING_ERROR")

    @staticmethod
    def max_lag_error(module: ModuleType) -> float:
        assert hasattr(module, "MAX_LAG_ERROR"), (
            f"Module {module.__name__} must define a MAX_LAG_ERROR "
            f"variable specifying the maximum allowed lag error in meters"
        )

        return getattr(module, "MAX_LAG_ERROR")


def load(module_path: str) -> ExampleModule:
    module = import_module(module_path)

    return ExampleModule(
        seed=extract.seed(module),
        reference=extract.reference(module),
        goal_fraction=extract.goal_fraction(module),
        max_contouring_error=extract.max_contouring_error(module),
        max_lag_error=extract.max_lag_error(module),
        create=extract.component_creator(module),
        run=extract.simulation_runner(module),
    )


@mark.parametrize(
    ["module_path"],
    [doc_example(path) for path in discover()],
    ids=[extract.module_id(path) for path in discover()],
)
@mark.visualize.with_args(visualizer.mpcc(), lambda seed: seed)
@mark.filterwarnings("ignore:.*'obstacle_states'.*not.*data.*")
@mark.integration
def test_that_documentation_example_produces_valid_plan(
    visualization: VisualizationData[MpccSimulationResult], module_path: str
) -> None:
    example = load(module_path)

    components = example.create()
    result = example.run(*components)

    visualization.data_is(result.visualization).seed_is(example.seed)

    assert result.reached_goal, (
        f"Vehicle did not make sufficient progress along the path. "
        f"Final progress: {result.progress:.1f}, "
        f"expected >= {example.reference.path_length * example.goal_fraction:.1f}"
    )

    assert (
        result.visualization.contouring_errors.max() < example.max_contouring_error
    ), (
        f"Max contouring error {result.visualization.contouring_errors.max():.2f} m "
        f"exceeds threshold {example.max_contouring_error:.2f} m"
    )

    assert result.visualization.lag_errors.max() < example.max_lag_error, (
        f"Max lag error {result.visualization.lag_errors.max():.2f} m "
        f"exceeds threshold {example.max_lag_error:.2f} m"
    )
