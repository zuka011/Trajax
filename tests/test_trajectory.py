from trajax import trajectory, types, Trajectory, PathParameters, ReferencePoints

import numpy as np
import jax.numpy as jnp
from numtypes import array

from pytest import mark


@mark.parametrize(
    ["trajectory", "path_parameters", "expected"],
    [
        (
            trajectory.numpy.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10),
            path_parameters := types.numpy.path_parameters(
                array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]], shape=(T := 3, M := 2))
            ),
            expected := types.numpy.reference_points(
                x=array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]], shape=(T, M)),
                y=array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], shape=(T, M)),
                heading=array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], shape=(T, M)),
            ),
        ),
        (
            trajectory.numpy.line(start=(0.0, 0.0), end=(3.0, 4.0), path_length=5),
            path_parameters := types.numpy.path_parameters(
                array([[0.0, 5.0], [2.5, 2.5], [5.0, 0.0]], shape=(T := 3, M := 2))
            ),
            expected := types.numpy.reference_points(
                x=array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0]], shape=(T, M)),
                y=array([[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]], shape=(T, M)),
                heading=array(
                    np.full((T, M), np.arctan2(4.0, 3.0)).tolist(), shape=(T, M)
                ),
            ),
        ),
        (
            trajectory.jax.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10),
            path_parameters := types.jax.path_parameters(
                array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]], shape=(T := 3, M := 2))
            ),
            expected := types.jax.reference_points(
                x=array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]], shape=(T, M)),
                y=array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], shape=(T, M)),
                heading=array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], shape=(T, M)),
            ),
        ),
        (
            trajectory.jax.line(start=(0.0, 0.0), end=(3.0, 4.0), path_length=5),
            path_parameters := types.jax.path_parameters(
                jnp.array([[0.0, 5.0], [2.5, 2.5], [5.0, 0.0]]),
                horizon=(T := 3),
                rollout_count=(M := 2),
            ),
            expected := types.jax.reference_points(
                x=jnp.array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0]]),
                y=jnp.array([[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]]),
                heading=jnp.array(jnp.full((T, M), jnp.arctan2(4.0, 3.0)).tolist()),
                horizon=T,
                rollout_count=M,
            ),
        ),
    ],
)
def test_that_batch_query_returns_correct_positions_and_headings[
    PathParametersT: PathParameters,
    ReferencePointsT: ReferencePoints,
](
    trajectory: Trajectory[PathParametersT, ReferencePointsT],
    path_parameters: PathParametersT,
    expected: ReferencePointsT,
) -> None:
    actual = trajectory.query(path_parameters)

    assert np.allclose(actual, expected)


@mark.parametrize(
    ["trajectory", "path_parameters", "expected"],
    [
        (  # Horizontal line from (0,0) to (10,0)
            trajectory.numpy.waypoints(
                points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)), path_length=5.0
            ),
            types.numpy.path_parameters(array([[0.0], [2.5], [5.0]], shape=(3, 1))),
            types.numpy.reference_points(
                x=array([[0.0], [5.0], [10.0]], shape=(3, 1)),
                y=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                heading=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
            ),
        ),
        (  # Vertical line from (0,0) to (0,10)
            trajectory.numpy.waypoints(
                points=array([[0.0, 0.0], [0.0, 10.0]], shape=(2, 2)), path_length=20.0
            ),
            types.numpy.path_parameters(array([[0.0], [10.0], [20.0]], shape=(3, 1))),
            types.numpy.reference_points(
                x=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                y=array([[0.0], [5.0], [10.0]], shape=(3, 1)),
                heading=array([[np.pi / 2], [np.pi / 2], [np.pi / 2]], shape=(3, 1)),
            ),
        ),
        (
            trajectory.jax.waypoints(
                points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)), path_length=5.0
            ),
            types.jax.path_parameters(array([[0.0], [2.5], [5.0]], shape=(3, 1))),
            types.jax.reference_points(
                x=array([[0.0], [5.0], [10.0]], shape=(3, 1)),
                y=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                heading=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
            ),
        ),
        (
            trajectory.jax.waypoints(
                points=jnp.array([[0.0, 0.0], [10.0, 0.0]]), path_length=20.0
            ),
            types.jax.path_parameters(
                jnp.array([[0.0], [10.0], [20.0]]),
                horizon=3,
                rollout_count=1,
            ),
            types.jax.reference_points(
                x=jnp.array([[0.0], [5.0], [10.0]]),
                y=jnp.array([[0.0], [0.0], [0.0]]),
                heading=jnp.array([[0.0], [0.0], [0.0]]),
                horizon=3,
                rollout_count=1,
            ),
        ),
    ],
)
def test_that_waypoints_interpolate_linearly_when_waypoints_follow_a_straight_line[
    PathParametersT: PathParameters,
    ReferencePointsT: ReferencePoints,
](
    trajectory: Trajectory[PathParametersT, ReferencePointsT],
    path_parameters: PathParametersT,
    expected: ReferencePointsT,
) -> None:
    actual = trajectory.query(path_parameters)

    assert np.allclose(actual, expected, atol=1e-6)


@mark.parametrize(
    ["trajectory", "path_parameters"],
    [
        (
            trajectory.numpy.waypoints(
                points=array([[1.0, 2.0], [5.0, 3.0], [10.0, 7.0]], shape=(3, 2)),
                path_length=20.0,
            ),
            types.numpy.path_parameters(array([[0.0]], shape=(1, 1))),
        ),
        (
            trajectory.jax.waypoints(
                points=array([[1.0, 2.0], [5.0, 3.0], [10.0, 7.0]], shape=(3, 2)),
                path_length=20.0,
            ),
            types.jax.path_parameters(array([[0.0]], shape=(1, 1))),
        ),
    ],
)
def test_that_query_returns_first_waypoint_when_path_parameter_is_zero[
    PathParametersT: PathParameters
](
    trajectory: Trajectory[PathParametersT, ReferencePoints],
    path_parameters: PathParametersT,
) -> None:
    actual = trajectory.query(path_parameters)

    assert np.allclose(actual.x(), 1.0, atol=1e-6)
    assert np.allclose(actual.y(), 2.0, atol=1e-6)


@mark.parametrize(
    ["trajectory", "path_parameters"],
    [
        (
            trajectory.numpy.waypoints(
                points=array([[0.0, 0.0], [3.0, 4.0]], shape=(2, 2)),
                path_length=(total_length := 7.2),
            ),
            types.numpy.path_parameters(array([[total_length]], shape=(1, 1))),
        ),
        (
            trajectory.jax.waypoints(
                points=array([[0.0, 0.0], [3.0, 4.0]], shape=(2, 2)),
                path_length=(total_length := 2.1),
            ),
            types.jax.path_parameters(array([[total_length]], shape=(1, 1))),
        ),
    ],
)
def test_that_query_returns_last_waypoint_when_path_parameter_is_total_length[
    PathParametersT: PathParameters,
](
    trajectory: Trajectory[PathParametersT, ReferencePoints],
    path_parameters: PathParametersT,
) -> None:
    actual = trajectory.query(path_parameters)

    assert np.allclose(actual.x(), 3.0, atol=1e-6)
    assert np.allclose(actual.y(), 4.0, atol=1e-6)


@mark.parametrize(
    ["trajectory", "path_parameters", "expected_heading"],
    [
        (  # Horizontal line (heading = 0)
            trajectory.numpy.waypoints(
                points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)), path_length=20.0
            ),
            types.numpy.path_parameters(array([[0.0], [10.0], [20.0]], shape=(3, 1))),
            expected_heading := 0.0,
        ),
        (  # Vertical line (heading = π/2)
            trajectory.numpy.waypoints(
                points=array([[0.0, 0.0], [0.0, 10.0]], shape=(2, 2)), path_length=1.0
            ),
            types.numpy.path_parameters(array([[0.0], [0.5], [1.0]], shape=(3, 1))),
            expected_heading := np.pi / 2,
        ),
        (  # Diagonal line at 45 degrees (heading = π/4)
            trajectory.numpy.waypoints(
                points=array([[0.0, 0.0], [10.0, 10.0]], shape=(2, 2)),
                path_length=1.0,
            ),
            types.numpy.path_parameters(array([[0.0], [0.5], [1.0]], shape=(3, 1))),
            expected_heading := np.pi / 4,
        ),
        (  # Diagonal line at 135 degrees (heading = 3π/4)
            trajectory.numpy.waypoints(
                points=array([[0.0, 0.0], [-10.0, 10.0]], shape=(2, 2)),
                path_length=1.0,
            ),
            types.numpy.path_parameters(array([[0.0], [0.5], [1.0]], shape=(3, 1))),
            expected_heading := 3 * np.pi / 4,
        ),
        (
            trajectory.jax.waypoints(
                points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)), path_length=1.0
            ),
            types.jax.path_parameters(array([[0.0], [0.5], [1.0]], shape=(3, 1))),
            expected_heading := 0.0,
        ),
        (
            trajectory.jax.waypoints(
                points=array([[0.0, 0.0], [0.0, 10.0]], shape=(2, 2)), path_length=1.0
            ),
            types.jax.path_parameters(array([[0.0], [0.5], [1.0]], shape=(3, 1))),
            expected_heading := np.pi / 2,
        ),
        (  # Diagonal line at -45 degrees (heading = -π/4)
            trajectory.jax.waypoints(
                points=jnp.array([[0.0, 0.0], [10.0, -10.0]]),
                path_length=10.0,
            ),
            types.jax.path_parameters(
                jnp.array([[0.0], [5.0], [10.0]]),
                horizon=3,
                rollout_count=1,
            ),
            expected_heading := -np.pi / 4,
        ),
    ],
)
def test_that_heading_matches_tangent_direction_of_waypoint_trajectory[
    PathParametersT: PathParameters,
](
    trajectory: Trajectory[PathParametersT, ReferencePoints],
    path_parameters: PathParametersT,
    expected_heading: float,
) -> None:
    actual = trajectory.query(path_parameters)

    assert np.allclose(actual.heading(), expected_heading, atol=1e-6)


@mark.parametrize(
    [
        "trajectory",
        "path_parameters",
        "expected_start_heading",
        "expected_end_heading",
        "should_increase",
    ],
    [
        (  # East -> North
            trajectory.numpy.waypoints(
                points=array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], shape=(3, 2)),
                path_length=2.0,
            ),
            types.numpy.path_parameters(array([[0.0], [1.0], [2.0]], shape=(3, 1))),
            expected_start_heading := 0.0,
            expected_end_heading := np.pi / 2,
            should_increase := True,
        ),
        (  # North -> East
            trajectory.numpy.waypoints(
                points=array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], shape=(3, 2)),
                path_length=2.0,
            ),
            types.numpy.path_parameters(array([[0.0], [1.0], [2.0]], shape=(3, 1))),
            expected_start_heading := np.pi / 2,
            expected_end_heading := 0.0,
            should_increase := False,
        ),
        (
            trajectory.jax.waypoints(
                points=jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
                path_length=2.0,
            ),
            types.jax.path_parameters(
                jnp.array([[0.0], [1.0], [2.0]]),
                horizon=3,
                rollout_count=1,
            ),
            expected_start_heading := 0.0,
            expected_end_heading := np.pi / 2,
            should_increase := True,
        ),
        (
            trajectory.jax.waypoints(
                points=jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                path_length=2.0,
            ),
            types.jax.path_parameters(
                jnp.array([[0.0], [1.0], [2.0]]),
                horizon=3,
                rollout_count=1,
            ),
            expected_start_heading := np.pi / 2,
            expected_end_heading := 0.0,
            should_increase := False,
        ),
    ],
)
def test_that_heading_changes_smoothly_through_waypoints[
    PathParametersT: PathParameters,
](
    trajectory: Trajectory[PathParametersT, ReferencePoints],
    path_parameters: PathParametersT,
    expected_start_heading: float,
    expected_end_heading: float,
    should_increase: bool,
) -> None:
    actual = trajectory.query(path_parameters)
    headings = actual.heading().flatten()

    assert np.allclose(headings[0], expected_start_heading, atol=0.2)
    assert np.allclose(headings[-1], expected_end_heading, atol=0.2)

    if should_increase:
        assert all(headings[i + 1] >= headings[i] for i in range(len(headings) - 1)), (
            f"Headings did not increase monotonically. Got: {headings}"
        )
    else:
        assert all(headings[i + 1] <= headings[i] for i in range(len(headings) - 1)), (
            f"Headings did not decrease monotonically. Got: {headings}"
        )


# TODO: Add tests for optimal behavior when following looped trajectories.
