from typing import Sequence

from trajax import (
    Trajectory,
    PathParameters,
    ReferencePoints,
    Positions,
    LateralPositions,
    LongitudinalPositions,
    trajectory,
    types,
)

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
                    array([[0.0], [2.5], [5.0]], shape=(T := 3, M := 1))
                ),
                expected := types.reference_points(
                    x=array([[0.0], [5.0], [10.0]], shape=(T, M)),
                    y=array([[0.0], [0.0], [0.0]], shape=(T, M)),
                    heading=array([[0.0], [0.0], [0.0]], shape=(T, M)),
                ),
            ),
            (  # Vertical line from (0,0) to (0,10)
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [0.0, 10.0]], shape=(2, 2)),
                    path_length=20.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [10.0], [20.0]], shape=(T := 3, M := 1))
                ),
                expected := types.reference_points(
                    x=array([[0.0], [0.0], [0.0]], shape=(T, M)),
                    y=array([[0.0], [5.0], [10.0]], shape=(T, M)),
                    heading=array(
                        [[np.pi / 2], [np.pi / 2], [np.pi / 2]], shape=(T, M)
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
                path_parameters := types.path_parameters(
                    array([[0.0]], shape=(T := 1, M := 1))
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
                    array([[total_length]], shape=(T := 1, M := 1))
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
                    array([[0.0], [10.0], [20.0]], shape=(T := 3, M := 1))
                ),
                expected_heading := 0.0,
            ),
            (  # Vertical line (heading = π/2)
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [0.0, 10.0]], shape=(2, 2)),
                    path_length=1.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [0.5], [1.0]], shape=(T := 3, M := 1))
                ),
                expected_heading := np.pi / 2,
            ),
            (  # Diagonal line at 45 degrees (heading = π/4)
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [10.0, 10.0]], shape=(2, 2)),
                    path_length=1.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [0.5], [1.0]], shape=(T := 3, M := 1))
                ),
                expected_heading := np.pi / 4,
            ),
            (  # Diagonal line at 135 degrees (heading = 3π/4)
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [-10.0, 10.0]], shape=(2, 2)),
                    path_length=1.0,
                ),
                path_parameters := types.path_parameters(
                    array([[0.0], [0.5], [1.0]], shape=(T := 3, M := 1))
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
                    array([[0.0], [1.0], [2.0]], shape=(T := 3, M := 1))
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
                    array([[0.0], [1.0], [2.0]], shape=(T := 3, M := 1))
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


class test_that_lateral_position_is_signed_perpendicular_distance_from_trajectory:
    @staticmethod
    def cases(trajectory, types) -> Sequence[tuple]:
        cases = []

        for horizontal_line in [
            trajectory.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0),
            trajectory.waypoints(
                points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)), path_length=10.0
            ),
        ]:
            cases.extend(
                [
                    (  # Point on horizontal line has zero lateral deviation
                        horizontal_line,
                        positions := types.positions(
                            x=array([[5.0]], shape=(T := 1, M := 1)),
                            y=array([[0.0]], shape=(T, M)),
                        ),
                        expected := types.lateral_positions(
                            array([[0.0]], shape=(T, M)),
                        ),
                    ),
                    (  # Point to the left of the horizontal line has negative lateral deviation
                        horizontal_line,
                        positions := types.positions(
                            x=array([[5.0]], shape=(T := 1, M := 1)),
                            y=array([[2.0]], shape=(T, M)),
                        ),
                        expected := types.lateral_positions(
                            array([[-2.0]], shape=(T, M)),
                        ),
                    ),
                    (  # Point to the right of the horizontal line has positive lateral deviation
                        horizontal_line,
                        positions := types.positions(
                            x=array([[5.0]], shape=(T := 1, M := 1)),
                            y=array([[-3.0]], shape=(T, M)),
                        ),
                        expected := types.lateral_positions(
                            array([[3.0]], shape=(T, M)),
                        ),
                    ),
                ]
            )

        for diagonal_line in [  # Diagonal trajectory 45 degrees
            trajectory.line(start=(0.0, 0.0), end=(10.0, 10.0), path_length=10.0),
            trajectory.waypoints(
                points=array([[0.0, 0.0], [10.0, 10.0]], shape=(2, 2)), path_length=10.0
            ),
        ]:
            cases.extend(
                [
                    (
                        diagonal_line,
                        positions := types.positions(
                            x=array([[0.0]], shape=(T := 1, M := 1)),
                            y=array([[1.0]], shape=(T, M)),
                        ),
                        expected := types.lateral_positions(
                            array([[-np.sqrt(2) / 2]], shape=(T, M)),
                        ),
                    )
                ]
            )

        for curved_line in [  # Curved trajectory with three waypoints
            trajectory.waypoints(
                points=array([[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]], shape=(3, 2)),
                path_length=10.0,
            )
        ]:
            no_deviation = types.lateral_positions(
                array([[0.0]], shape=(T := 1, M := 1))
            )
            cases.extend(
                [
                    (  # First waypoint
                        curved_line,
                        positions := types.positions(
                            x=array([[0.0]], shape=(T := 1, M := 1)),
                            y=array([[0.0]], shape=(T, M)),
                        ),
                        expected := no_deviation,
                    ),
                    (  # Last waypoint
                        curved_line,
                        positions := types.positions(
                            x=array([[10.0]], shape=(T := 1, M := 1)),
                            y=array([[0.0]], shape=(T, M)),
                        ),
                        expected := no_deviation,
                    ),
                    (  # Middle waypoint
                        curved_line,
                        positions := types.positions(
                            x=array([[5.0]], shape=(T := 1, M := 1)),
                            y=array([[5.0]], shape=(T, M)),
                        ),
                        expected := no_deviation,
                    ),
                    (
                        # Left of middle waypoint
                        curved_line,
                        positions := types.positions(
                            x=array([[5.0]], shape=(T := 1, M := 1)),
                            y=array([[6.0]], shape=(T, M)),
                        ),
                        expected := types.lateral_positions(
                            array([[-1.0]], shape=(T, M))
                        ),
                    ),
                ]
            )

        for l_shaped_line in [  # L-shaped trajectory with three waypoints
            trajectory.waypoints(
                points=array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], shape=(3, 2)),
                path_length=20.0,
            )
        ]:
            cases.extend(
                [
                    (  # Point diagonally offset left from middle point
                        l_shaped_line,
                        positions := types.positions(
                            x=array([[9.0]], shape=(T := 1, M := 1)),
                            y=array([[1.0]], shape=(T, M)),
                        ),
                        expected := types.lateral_positions(
                            array([[-1.0 * np.sqrt(2)]], shape=(T, M))
                        ),
                    ),
                    (  # Point diagonally offset right from middle point
                        l_shaped_line,
                        positions := types.positions(
                            x=array([[12.0]], shape=(T := 1, M := 1)),
                            y=array([[-2.0]], shape=(T, M)),
                        ),
                        expected := types.lateral_positions(
                            array([[2.0 * np.sqrt(2)]], shape=(T, M))
                        ),
                    ),
                    (  # Multi-step, Multi-rollout query
                        l_shaped_line,
                        positions := types.positions(
                            x=array(
                                [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
                                shape=(T := 3, M := 2),
                            ),
                            y=array(
                                [[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]], shape=(T, M)
                            ),
                        ),
                        expected := types.lateral_positions(
                            array(
                                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                                * np.sqrt(2),
                                shape=(T, M),
                            )
                        ),
                    ),
                ]
            )

        return cases

    @mark.parametrize(
        ["trajectory", "positions", "expected"],
        [
            *cases(trajectory=trajectory.numpy, types=types.numpy),
            *cases(trajectory=trajectory.jax, types=types.jax),
        ],
    )
    def test[PositionsT: Positions, LateralT: LateralPositions](
        self,
        trajectory: Trajectory[
            PathParameters, ReferencePoints, PositionsT, LateralT, LongitudinalPositions
        ],
        positions: PositionsT,
        expected: LateralT,
    ) -> None:
        actual = trajectory.lateral(positions)

        assert np.allclose(actual, expected, atol=1e-6)


class test_that_longitudinal_position_is_distance_along_trajectory:
    @staticmethod
    def cases(trajectory, types) -> Sequence[tuple]:
        cases = []

        for horizontal_line in [
            trajectory.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0),
            trajectory.waypoints(
                points=array([[0.0, 0.0], [10.0, 0.0]], shape=(2, 2)), path_length=10.0
            ),
        ]:
            cases.extend(
                [
                    (  # Point at start of horizontal line
                        trajectory.line(
                            start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0
                        ),
                        positions := types.positions(
                            x=array([[0.0]], shape=(T := 1, M := 1)),
                            y=array([[0.0]], shape=(T, M)),
                        ),
                        expected := types.longitudinal_positions(
                            array([[0.0]], shape=(T, M)),
                        ),
                    ),
                    (  # Point at middle of horizontal line
                        trajectory.line(
                            start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0
                        ),
                        positions := types.positions(
                            x=array([[5.0]], shape=(T := 1, M := 1)),
                            y=array([[0.0]], shape=(T, M)),
                        ),
                        expected := types.longitudinal_positions(
                            array([[5.0]], shape=(T, M)),
                        ),
                    ),
                    (  # Point at end of horizontal line
                        trajectory.line(
                            start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0
                        ),
                        positions := types.positions(
                            x=array([[10.0]], shape=(T := 1, M := 1)),
                            y=array([[0.0]], shape=(T, M)),
                        ),
                        expected := types.longitudinal_positions(
                            array([[10.0]], shape=(T, M)),
                        ),
                    ),
                    (  # Point offset from horizontal line (projection matters)
                        trajectory.line(
                            start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0
                        ),
                        positions := types.positions(
                            x=array([[5.0]], shape=(T := 1, M := 1)),
                            y=array([[3.0]], shape=(T, M)),
                        ),
                        expected := types.longitudinal_positions(
                            array([[5.0]], shape=(T, M)),
                        ),
                    ),
                ]
            )

        for diagonal_line in [  # Diagonal trajectory 45 degrees
            trajectory.line(start=(0.0, 0.0), end=(10.0, 10.0), path_length=10.0),
            trajectory.waypoints(
                points=array([[0.0, 0.0], [10.0, 10.0]], shape=(2, 2)), path_length=10.0
            ),
        ]:
            cases.extend(
                [
                    (
                        trajectory.line(
                            start=(0.0, 0.0), end=(10.0, 10.0), path_length=10.0
                        ),
                        positions := types.positions(
                            x=array([[5.0]], shape=(T := 1, M := 1)),
                            y=array([[5.0]], shape=(T, M)),
                        ),
                        expected := types.longitudinal_positions(
                            array([[np.sqrt(50) / np.sqrt(200) * 10.0]], shape=(T, M)),
                        ),
                    ),
                ]
            )

        for curved_line in [  # Long curved trajectory with three waypoints
            trajectory.waypoints(
                points=array([[0.0, 0.0], [10.0, 5.0], [20.0, 0.0]], shape=(3, 2)),
                path_length=20.0,
            )
        ]:
            cases.extend(
                [
                    (  # Below middle waypoint
                        curved_line,
                        positions := types.positions(
                            x=array([[10.0]], shape=(1, 1)),
                            y=array([[0.0]], shape=(1, 1)),
                        ),
                        expected := types.lateral_positions(
                            array([[10.0]], shape=(1, 1))
                        ),
                    ),
                    (
                        # Above middle waypoint
                        curved_line,
                        positions := types.positions(
                            x=array([[10.0]], shape=(1, 1)),
                            y=array([[30.0]], shape=(1, 1)),
                        ),
                        expected := types.lateral_positions(
                            array([[10.0]], shape=(1, 1))
                        ),
                    ),
                    (  # Before first waypoint
                        curved_line,
                        positions := types.positions(
                            x=array([[-5.0]], shape=(1, 1)),
                            y=array([[0.0]], shape=(1, 1)),
                        ),
                        expected := types.lateral_positions(
                            array([[0.0]], shape=(1, 1))
                        ),
                    ),
                    (  # After last waypoint
                        curved_line,
                        positions := types.positions(
                            x=array([[25.0]], shape=(1, 1)),
                            y=array([[0.0]], shape=(1, 1)),
                        ),
                        expected := types.lateral_positions(
                            array([[20.0]], shape=(1, 1))
                        ),
                    ),
                    (  # Multi-step, Multi-rollout query
                        curved_line,
                        positions := types.positions(
                            x=array(
                                [[10.0, -5.0], [25.0, 10.0], [-10.0, 30.0]],
                                shape=(T := 3, M := 2),
                            ),
                            y=array(
                                [[0.0, -5.0], [-5.0, 10.0], [5.0, 5.0]], shape=(T, M)
                            ),
                        ),
                        expected := types.longitudinal_positions(
                            array(
                                np.array([[10.0, 0.0], [20.0, 10.0], [0.0, 20.0]]),
                                shape=(T, M),
                            )
                        ),
                    ),
                ]
            )

        return cases

    @mark.parametrize(
        ["trajectory", "positions", "expected"],
        [
            *cases(trajectory=trajectory.numpy, types=types.numpy),
            *cases(trajectory=trajectory.jax, types=types.jax),
        ],
    )
    def test[PositionsT: Positions, LongitudinalT: LongitudinalPositions](
        self,
        trajectory: Trajectory[
            PathParameters, ReferencePoints, PositionsT, LateralPositions, LongitudinalT
        ],
        positions: PositionsT,
        expected: LongitudinalT,
    ) -> None:
        actual = trajectory.longitudinal(positions)

        assert np.allclose(actual, expected, atol=1e-5)


# TODO: Add tests for optimal behavior when following looped trajectories.
