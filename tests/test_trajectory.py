from typing import Sequence

from trajax import trajectory, types, Trajectory, PathParameters, ReferencePoints

from numtypes import array

import numpy as np

from pytest import mark


class test_that_batch_query_returns_correct_positions_and_headings:
    @staticmethod
    def cases(trajectory, types) -> Sequence[tuple]:
        return [
            (
                trajectory.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10),
                path_parameters := types.path_parameters(
                    array(
                        [[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]],
                        shape=(T := 3, M := 2),
                    )
                ),
                expected := types.reference_points(
                    x=array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0]], shape=(T, M)),
                    y=array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], shape=(T, M)),
                    heading=array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], shape=(T, M)),
                ),
            ),
            (
                trajectory.line(start=(0.0, 0.0), end=(3.0, 4.0), path_length=5),
                path_parameters := types.path_parameters(
                    array([[0.0, 5.0], [2.5, 2.5], [5.0, 0.0]], shape=(T := 3, M := 2))
                ),
                expected := types.reference_points(
                    x=array([[0.0, 3.0], [1.5, 1.5], [3.0, 0.0]], shape=(T, M)),
                    y=array([[0.0, 4.0], [2.0, 2.0], [4.0, 0.0]], shape=(T, M)),
                    heading=array(
                        np.full((T, M), np.arctan2(4.0, 3.0)).tolist(), shape=(T, M)
                    ),
                ),
            ),
        ]

    @mark.parametrize(
        ["trajectory", "path_parameters", "expected"],
        [
            *cases(trajectory=trajectory.numpy, types=types.numpy),
            *cases(trajectory=trajectory.jax, types=types.jax),
        ],
    )
    def test[PathParametersT: PathParameters, ReferencePointsT: ReferencePoints](
        self,
        trajectory: Trajectory[PathParametersT, ReferencePointsT],
        path_parameters: PathParametersT,
        expected: ReferencePointsT,
    ) -> None:
        actual = trajectory.query(path_parameters)

        assert np.allclose(actual, expected)


class test_that_waypoints_interpolate_linearly_when_waypoints_follow_a_straight_line:
    @staticmethod
    def cases(trajectory, types) -> Sequence[tuple]:
        return [
            (  # Horizontal line from (0,0) to (10,0)
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)),
                    path_length=5.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [2.5], [5.0]], shape=(3, 1))
                ),
                expected := types.reference_points(
                    x=array([[0.0], [5.0], [10.0]], shape=(3, 1)),
                    y=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                    heading=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                ),
            ),
            (  # Vertical line from (0,0) to (0,10)
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [0.0, 10.0]], shape=(2, 2)),
                    path_length=20.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [10.0], [20.0]], shape=(3, 1))
                ),
                expected := types.reference_points(
                    x=array([[0.0], [0.0], [0.0]], shape=(3, 1)),
                    y=array([[0.0], [5.0], [10.0]], shape=(3, 1)),
                    heading=array(
                        [[np.pi / 2], [np.pi / 2], [np.pi / 2]], shape=(3, 1)
                    ),
                ),
            ),
        ]

    @mark.parametrize(
        ["trajectory", "path_parameters", "expected"],
        [
            *cases(trajectory=trajectory.numpy, types=types.numpy),
            *cases(trajectory=trajectory.jax, types=types.jax),
        ],
    )
    def test[PathParametersT: PathParameters, ReferencePointsT: ReferencePoints](
        self,
        trajectory: Trajectory[PathParametersT, ReferencePointsT],
        path_parameters: PathParametersT,
        expected: ReferencePointsT,
    ) -> None:
        actual = trajectory.query(path_parameters)

        assert np.allclose(actual, expected, atol=1e-6)


class test_that_query_returns_first_waypoint_when_path_parameter_is_zero:
    @staticmethod
    def cases(trajectory, types) -> Sequence[tuple]:
        return [
            (
                trajectory.waypoints(
                    points=array([[1.0, 2.0], [5.0, 3.0], [10.0, 7.0]], shape=(3, 2)),
                    path_length=20.0,
                ),
                path_parameters := types.path_parameters(array([[0.0]], shape=(1, 1))),
            ),
        ]

    @mark.parametrize(
        ["trajectory", "path_parameters"],
        [
            *cases(trajectory=trajectory.numpy, types=types.numpy),
            *cases(trajectory=trajectory.jax, types=types.jax),
        ],
    )
    def test[PathParametersT: PathParameters](
        self,
        trajectory: Trajectory[PathParametersT, ReferencePoints],
        path_parameters: PathParametersT,
    ) -> None:
        actual = trajectory.query(path_parameters)

        assert np.allclose(actual.x(), 1.0, atol=1e-6)
        assert np.allclose(actual.y(), 2.0, atol=1e-6)


class test_that_query_returns_last_waypoint_when_path_parameter_is_total_length:
    @staticmethod
    def cases(trajectory, types) -> Sequence[tuple]:
        return [
            (
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [3.0, 4.0]], shape=(2, 2)),
                    path_length=(total_length := 7.2),
                ),
                path_parameters := types.path_parameters(
                    array([[total_length]], shape=(1, 1))
                ),
            ),
        ]

    @mark.parametrize(
        ["trajectory", "path_parameters"],
        [
            *cases(trajectory=trajectory.numpy, types=types.numpy),
            *cases(trajectory=trajectory.jax, types=types.jax),
        ],
    )
    def test[PathParametersT: PathParameters](
        self,
        trajectory: Trajectory[PathParametersT, ReferencePoints],
        path_parameters: PathParametersT,
    ) -> None:
        actual = trajectory.query(path_parameters)

        assert np.allclose(actual.x(), 3.0, atol=1e-6)
        assert np.allclose(actual.y(), 4.0, atol=1e-6)


class test_that_heading_matches_tangent_direction_of_waypoint_trajectory:
    @staticmethod
    def cases(trajectory, types) -> Sequence[tuple]:
        return [
            (  # Horizontal line (heading = 0)
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)),
                    path_length=20.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [10.0], [20.0]], shape=(3, 1))
                ),
                expected_heading := 0.0,
            ),
            (  # Vertical line (heading = π/2)
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [0.0, 10.0]], shape=(2, 2)),
                    path_length=1.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [0.5], [1.0]], shape=(3, 1))
                ),
                expected_heading := np.pi / 2,
            ),
            (  # Diagonal line at 45 degrees (heading = π/4)
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [10.0, 10.0]], shape=(2, 2)),
                    path_length=1.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [0.5], [1.0]], shape=(3, 1))
                ),
                expected_heading := np.pi / 4,
            ),
            (  # Diagonal line at 135 degrees (heading = 3π/4)
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [-10.0, 10.0]], shape=(2, 2)),
                    path_length=1.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [0.5], [1.0]], shape=(3, 1))
                ),
                expected_heading := 3 * np.pi / 4,
            ),
        ]

    @mark.parametrize(
        ["trajectory", "path_parameters", "expected_heading"],
        [
            *cases(trajectory=trajectory.numpy, types=types.numpy),
            *cases(trajectory=trajectory.jax, types=types.jax),
        ],
    )
    def test[PathParametersT: PathParameters](
        self,
        trajectory: Trajectory[PathParametersT, ReferencePoints],
        path_parameters: PathParametersT,
        expected_heading: float,
    ) -> None:
        actual = trajectory.query(path_parameters)

        assert np.allclose(actual.heading(), expected_heading, atol=1e-6)


class test_that_heading_changes_smoothly_through_waypoints:
    @staticmethod
    def cases(trajectory, types) -> Sequence[tuple]:
        return [
            (  # East -> North
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], shape=(3, 2)),
                    path_length=2.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [1.0], [2.0]], shape=(3, 1))
                ),
                expected_start_heading := 0.0,
                expected_end_heading := np.pi / 2,
                should_increase := True,
            ),
            (  # North -> East
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], shape=(3, 2)),
                    path_length=2.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [1.0], [2.0]], shape=(3, 1))
                ),
                expected_start_heading := np.pi / 2,
                expected_end_heading := 0.0,
                should_increase := False,
            ),
        ]

    @mark.parametrize(
        [
            "trajectory",
            "path_parameters",
            "expected_start_heading",
            "expected_end_heading",
            "should_increase",
        ],
        [
            *cases(trajectory=trajectory.numpy, types=types.numpy),
            *cases(trajectory=trajectory.jax, types=types.jax),
        ],
    )
    def test[PathParametersT: PathParameters](
        self,
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
            assert all(
                headings[i + 1] >= headings[i] for i in range(len(headings) - 1)
            ), f"Headings did not increase monotonically. Got: {headings}"
        else:
            assert all(
                headings[i + 1] <= headings[i] for i in range(len(headings) - 1)
            ), f"Headings did not decrease monotonically. Got: {headings}"


# TODO: Add tests for optimal behavior when following looped trajectories.
