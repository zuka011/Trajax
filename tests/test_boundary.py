from typing import Sequence

from faran import (
    BoundaryDistanceExtractor,
    ExplicitBoundary,
    types,
    trajectory,
    boundary,
)

from numtypes import Array, array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


class test_that_extractor_returns_zero_distance_when_ego_is_on_the_boundary:
    @staticmethod
    def cases(trajectory, types, data, create_boundary) -> Sequence[tuple]:
        return [
            (  # Ego on left boundary of line trajectory
                boundary := create_boundary.fixed_width(
                    reference=trajectory.line(
                        start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    left=(left_width := 2.0),
                    right=5.0,
                ),
                states := data.state_batch(
                    array(
                        # Left boundary at y = left_width
                        [[[x := 5.0], [y := left_width], [phi := 5.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                expected_distance := 0.0,
            ),
            (  # Ego on right boundary of line trajectory
                boundary := create_boundary.fixed_width(
                    reference=trajectory.line(
                        start=(0.0, 0.0),
                        end=(10.0, 0.0),
                        path_length=10.0,
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    left=2.0,
                    right=(right_width := 5.0),
                ),
                states := data.state_batch(
                    array(
                        # Right boundary at y = -right_width
                        [[[x := 5.0], [y := -right_width], [phi := 5.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                expected_distance := 0.0,
            ),
            (  # Ego on boundary at different longitudinal segments
                boundary := create_boundary.piecewise_fixed_width(
                    reference=trajectory.line(
                        start=(0.0, 0.0),
                        end=(10.0, 0.0),
                        path_length=10.0,
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    widths={
                        0.0: {"left": 2.0, "right": 4.0},
                        5.0: {"left": 3.0, "right": 5.0},
                        7.0: {"left": 1.0, "right": 2.0},
                    },
                ),
                states := data.state_batch(
                    array(
                        [
                            [
                                x := [2.5, 6.0, 9.0],
                                # Left boundary: y = left(s)
                                y := [2.0, 3.0, 1.0],
                                phi := [0.0, 0.0, 0.0],
                            ]
                        ],
                        shape=(T := 1, D_x := 3, M := 3),
                    )
                ),
                expected_distance := 0.0,
            ),
            (  # Segment boundary selection at s=5.0 should use the second segment
                boundary := create_boundary.piecewise_fixed_width(
                    reference=trajectory.line(
                        start=(0.0, 0.0),
                        end=(10.0, 0.0),
                        path_length=10.0,
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    widths={
                        0.0: {"left": 2.0, "right": 4.0},
                        5.0: {"left": 3.0, "right": 5.0},
                        7.0: {"left": 1.0, "right": 2.0},
                    },
                ),
                states := data.state_batch(
                    array(
                        [
                            [
                                [x := 5.0],
                                # Right boundary: y = -right(s); at s=5, right=5
                                [y := -5.0],
                                [phi := 0.0],
                            ]
                        ],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                expected_distance := 0.0,
            ),
        ]

    @mark.parametrize(
        ["boundary", "states", "expected_distance"],
        [
            *cases(
                trajectory=trajectory.numpy,
                types=types.numpy,
                data=data.numpy,
                create_boundary=boundary.numpy,
            ),
            *cases(
                trajectory=trajectory.jax,
                types=types.jax,
                data=data.jax,
                create_boundary=boundary.jax,
            ),
        ],
    )
    def test[StateBatchT](
        self,
        boundary: BoundaryDistanceExtractor,
        states: StateBatchT,
        expected_distance: float,
    ) -> None:
        distances = np.asarray(boundary(states=states))

        assert np.allclose(distances, expected_distance, atol=1e-6), (
            f"Boundary distance should be {expected_distance} when on boundary. "
            f"Got: {distances}"
        )


class test_that_extractor_returns_positive_distance_inside_boundary:
    @staticmethod
    def cases(trajectory, types, data, create_boundary) -> Sequence[tuple]:
        return [
            (  # Vehicle inside corridor on line trajectory
                boundary := create_boundary.fixed_width(
                    reference=trajectory.line(
                        start=(0.0, 0.0),
                        end=(10.0, 0.0),
                        path_length=10.0,
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    left=2.0,
                    right=5.0,
                ),
                states := data.state_batch(
                    array(
                        [
                            [  # Left 2m, left 1m, left 3m, right 3m, left/right 3.5m
                                x := [5.0, 5.0, 5.0, 6.0, 7.0],
                                y := [0.0, 1.0, -1.0, -2.0, -1.5],
                                phi := [5.0, 5.0, 5.0, 5.0, 5.0],
                            ]
                        ],
                        shape=(T := 1, D_x := 3, M := 5),
                    )
                ),
                expected_distances := array([[2.0, 1.0, 3.0, 3.0, 3.5]], shape=(T, M)),
            ),
            (  # Vehicle inside corridor with piecewise widths
                boundary := create_boundary.piecewise_fixed_width(
                    reference=trajectory.line(
                        start=(0.0, 0.0),
                        end=(10.0, 0.0),
                        path_length=10.0,
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    widths={
                        0.0: {"left": 2.0, "right": 4.0},
                        5.0: {"left": 3.0, "right": 5.0},
                        7.0: {"left": 1.0, "right": 2.0},
                    },
                ),
                states := data.state_batch(
                    array(
                        [
                            [
                                x := [1.0, 5.5, 8.0],
                                # Centerline (y=0) is always inside
                                y := [0.0, 0.0, 0.0],
                                phi := [0.0, 0.0, 0.0],
                            ]
                        ],
                        shape=(T := 1, D_x := 3, M := 3),
                    )
                ),
                expected_distances := array([[2.0, 3.0, 1.0]], shape=(T, M)),
            ),
        ]

    @mark.parametrize(
        ["boundary", "states", "expected_distances"],
        [
            *cases(
                trajectory=trajectory.numpy,
                types=types.numpy,
                data=data.numpy,
                create_boundary=boundary.numpy,
            ),
            *cases(
                trajectory=trajectory.jax,
                types=types.jax,
                data=data.jax,
                create_boundary=boundary.jax,
            ),
        ],
    )
    def test[StateBatchT](
        self,
        boundary: BoundaryDistanceExtractor,
        states: StateBatchT,
        expected_distances: np.ndarray,
    ) -> None:
        distances = np.asarray(boundary(states=states))

        assert np.allclose(distances, expected_distances, atol=1e-6), (
            f"Boundary distances do not match expected. "
            f"Expected: {expected_distances}, Got: {distances}"
        )


class test_that_extractor_returns_negative_distance_outside_boundary:
    @staticmethod
    def cases(trajectory, types, data, create_boundary) -> Sequence[tuple]:
        return [
            (
                boundary := create_boundary.fixed_width(
                    reference=trajectory.line(
                        start=(2.0, 0.0),
                        end=(2.0, -10.0),
                        path_length=10.0,
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    left=3.0,
                    right=4.0,
                ),
                states := data.state_batch(
                    array(
                        [
                            [
                                x := [-2.5, -4.0],
                                y := [-4.0, -3.0],
                                phi := [0.0, 0.0],
                            ],
                            [
                                x := [6.0, 10.0],
                                y := [-5.0, -8.0],
                                phi := [0.0, 0.0],
                            ],
                        ],
                        shape=(T := 2, D_x := 3, M := 2),
                    )
                ),
                expected_distance := array([[-0.5, -2.0], [-1.0, -5.0]], shape=(T, M)),
            ),
            (
                boundary := create_boundary.piecewise_fixed_width(
                    reference=trajectory.line(
                        start=(0.0, 0.0),
                        end=(10.0, 0.0),
                        path_length=10.0,
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    widths={
                        0.0: {"left": 2.0, "right": 4.0},
                        5.0: {"left": 3.0, "right": 5.0},
                        7.0: {"left": 1.0, "right": 2.0},
                    },
                ),
                states := data.state_batch(
                    array(
                        [
                            [
                                # Outside left in segment 1: y > left
                                x := [2.0],
                                y := [2.5],
                                phi := [0.0],
                            ],
                            [
                                # Outside right in segment 2: y < -right
                                x := [6.0],
                                y := [-6.0],
                                phi := [0.0],
                            ],
                            [
                                # Outside left in segment 3: y > left
                                x := [9.0],
                                y := [2.0],
                                phi := [0.0],
                            ],
                        ],
                        shape=(T := 3, D_x := 3, M := 1),
                    )
                ),
                expected_distance := array([[-0.5], [-1.0], [-1.0]], shape=(T, M)),
            ),
        ]

    @mark.parametrize(
        ["boundary", "states", "expected_distance"],
        [
            *cases(
                trajectory=trajectory.numpy,
                types=types.numpy,
                data=data.numpy,
                create_boundary=boundary.numpy,
            ),
            *cases(
                trajectory=trajectory.jax,
                types=types.jax,
                data=data.jax,
                create_boundary=boundary.jax,
            ),
        ],
    )
    def test[StateBatchT](
        self,
        boundary: BoundaryDistanceExtractor,
        states: StateBatchT,
        expected_distance: Array,
    ) -> None:
        distances = np.asarray(boundary(states=states))

        assert np.allclose(distances, expected_distance, atol=1e-6), (
            f"Boundary distance should be {expected_distance} when outside corridor. "
            f"Got: {distances}"
        )


class test_that_distance_between_edges_of_explicit_boundary_is_constant_when_width_is_fixed:
    @staticmethod
    def cases(trajectory, types, create_boundary) -> Sequence[tuple]:
        return [
            (
                boundary := create_boundary.fixed_width(
                    reference=reference,
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    left=(left_width := 3.0),
                    right=(right_width := 4.0),
                ),
                left_width,
                right_width,
            )
            for reference in (
                trajectory.line(
                    start=(0.0, 0.0),
                    end=(10.0, 0.0),
                    path_length=10.0,
                ),
                trajectory.line(
                    start=(5.0, 5.0),
                    end=(5.0, -5.0),
                    path_length=10.0,
                ),
                trajectory.waypoints(
                    points=array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]], shape=(3, 2)),
                    path_length=20.0,
                ),
            )
        ]

    @mark.parametrize(
        ["boundary", "left_width", "right_width"],
        [
            *cases(
                trajectory=trajectory.numpy,
                types=types.numpy,
                create_boundary=boundary.numpy,
            ),
            *cases(
                trajectory=trajectory.jax, types=types.jax, create_boundary=boundary.jax
            ),
        ],
    )
    def test(
        self, boundary: ExplicitBoundary, left_width: float, right_width: float
    ) -> None:
        left_boundary_points = boundary.left(sample_count=(N := 100))
        right_boundary_points = boundary.right(sample_count=N)

        distances = np.linalg.norm(left_boundary_points - right_boundary_points, axis=1)

        assert len(distances) == N
        assert np.allclose(
            distances, expected := left_width + right_width, atol=1e-6
        ), (
            f"Distance between left and right boundary points should be constant "
            f"and equal to the total width {expected}. "
            f"Got distances: {distances}"
        )


class test_that_distance_between_edges_of_explicit_boundary_matches_piecewise_widths:
    @staticmethod
    def cases(trajectory, types, create_boundary) -> Sequence[tuple]:
        return [
            (
                boundary := create_boundary.piecewise_fixed_width(
                    reference=trajectory.line(
                        start=(0.0, 0.0),
                        end=(10.0, 0.0),
                        path_length=10.0,
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0],
                        y=states.array[:, 1],
                    ),
                    widths={
                        0.0: {"left": 2.0, "right": 4.0},
                        5.0: {"left": 3.0, "right": 5.0},
                        7.0: {"left": 1.0, "right": 2.0},
                    },
                ),
                # Sample count = 5 -> s in {0.0, 2.5, 5.0, 7.5, 10.0}
                expected_widths := array([6.0, 6.0, 8.0, 3.0, 3.0], shape=(5,)),
            )
        ]

    @mark.parametrize(
        ["boundary", "expected_widths"],
        [
            *cases(
                trajectory=trajectory.numpy,
                types=types.numpy,
                create_boundary=boundary.numpy,
            ),
            *cases(
                trajectory=trajectory.jax,
                types=types.jax,
                create_boundary=boundary.jax,
            ),
        ],
    )
    def test(self, boundary: ExplicitBoundary, expected_widths: Array) -> None:
        left_boundary_points = boundary.left(sample_count=(N := 5))
        right_boundary_points = boundary.right(sample_count=N)

        distances = np.linalg.norm(left_boundary_points - right_boundary_points, axis=1)

        assert len(distances) == N
        assert np.allclose(distances, expected_widths, atol=1e-6), (
            f"Distance between left and right boundary points should match the "
            f"piecewise corridor widths. Expected: {expected_widths}, Got: {distances}"
        )
