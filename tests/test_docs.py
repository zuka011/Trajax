"""Documentation examples — parametrised test that runs each example script
and validates basic planning outcomes (goal reached, no collision, bounded
tracking errors)."""

from importlib import import_module
from types import ModuleType

from trajax_visualizer import MpccSimulationResult, visualizer

from tests.utilities import VisualizationData, doc_example
from pytest import mark


EXAMPLES = [
    "docs.examples.01_basic_path_following",
    "docs.examples.02_path_following_with_boundaries",
    "docs.examples.03_obstacle_avoidance",
    "docs.examples.04_obstacle_avoidance_with_uncertainty",
]


def _load(module_path: str) -> ModuleType:
    return import_module(module_path)


@mark.parametrize(
    ["module_path"],
    [doc_example(path) for path in EXAMPLES],
    ids=[path.rsplit(".", 1)[-1] for path in EXAMPLES],
)
@mark.visualize.with_args(visualizer.mpcc(), lambda seed: seed)
@mark.filterwarnings("ignore:.*'obstacle_states'.*not.*data.*")
@mark.integration
def test_that_documentation_example_produces_valid_plan(
    visualization: VisualizationData[MpccSimulationResult],
    module_path: str,
) -> None:
    example = _load(module_path)

    components = example.create()
    result = example.run(*components)

    visualization.data_is(result.visualization).seed_is(example.SEED)

    goal_fraction = getattr(example, "GOAL_FRACTION", 0.9)
    assert result.reached_goal, (
        f"Vehicle did not make sufficient progress along the path. "
        f"Final progress: {result.progress:.1f}, "
        f"expected >= {example.REFERENCE.path_length * goal_fraction:.1f}"
    )

    assert (
        result.visualization.contouring_errors.max() < example.MAX_CONTOURING_ERROR
    ), (
        f"Max contouring error {result.visualization.contouring_errors.max():.2f} m "
        f"exceeds threshold {example.MAX_CONTOURING_ERROR:.2f} m"
    )

    assert result.visualization.lag_errors.max() < example.MAX_LAG_ERROR, (
        f"Max lag error {result.visualization.lag_errors.max():.2f} m "
        f"exceeds threshold {example.MAX_LAG_ERROR:.2f} m"
    )
