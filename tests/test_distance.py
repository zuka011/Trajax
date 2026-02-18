from typing import Sequence

from trajax import (
    types,
    Circles,
    ConvexPolygon,
    Distance,
    DistanceExtractor,
    StateBatch,
    ObstacleStates,
    distance,
)

from numtypes import array, Array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


class test_that_distance_is_computed_correctly_when_ego_and_obstacle_are_single_circles:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (  # Basic distance along x-axis
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 1.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[x := 0.0], [y := 0.0], [theta := 0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                # K = 1 Obstacle, N = 1 Sample
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[x_o := 5.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[y_o := 0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [[[[x_o - x - r - r_o]]]], shape=(T, V := 1, M, N)
                ),
            ),
            (  # Distance with diagonal (3-4-5 triangle)
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 1.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[x := 0.0], [y := 0.0], [theta := 0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[x_o := 3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[y_o := 4.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                # distance = sqrt(3^2 + 4^2) - 1 - 1 = 5 - 2 = 3
                expected_distances := array([[[[3.0]]]], shape=(T, V := 1, M, N)),
            ),
            (  # Penetration (negative distance)
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 1.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[x := 0.0], [y := 0.0], [theta := 0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[x_o := 1.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[y_o := 0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                # distance = 1 - 1 - 1 = -1 (penetration)
                expected_distances := array([[[[-1.0]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Distance should be computed correctly. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_that_distance_is_computed_correctly_when_ego_is_multiple_circles:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.circles(
                    ego=Circles(
                        origins=array(
                            [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], shape=(V := 3, 2)
                        ),
                        radii=array([0.5, 0.5, 0.5], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 1.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[10.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [[[[8.5]], [[6.5]], [[4.5]]]], shape=(T, V := 3, M, N)
                ),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Each ego part should have its own distance. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_closest_obstacle_circle_is_used_when_obstacle_is_multiple_circles:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array(
                            [[10.0, 0.0], [5.0, 0.0], [7.0, 3.0]], shape=(C := 3, 2)
                        ),
                        radii=array([1.0, 0.5, 0.5], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[0.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                # Closest is circle at (5,0) with r=0.5: 5 - 1 - 0.5 = 3.5
                expected_distances := array([[[[3.5]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Should use closest obstacle circle. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_ego_circle_offsets_are_applied_when_ego_circles_are_not_centered_at_origin:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (  # Ego has two circles offset along x-axis
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(V := 2, 2)),
                        radii=array([0.5, 0.5], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 0.5], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    # Ego center at (0, 0)
                    # Circle 1 at (1, 0), Circle 2 at (-1, 0)
                    # Obstacle at (5, 0)
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[5.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [
                        [
                            [[3.0]],  # ||(5,0)-(1,0)|| - 1.0 = 4 - 1 = 3.0
                            [[5.0]],  # ||(5,0)-(-1,0)|| - 1.0 = 6 - 1 = 5.0
                        ]
                    ],
                    shape=(T, V := 2, M, N),
                ),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Ego circle offsets should be applied. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_obstacle_circle_offsets_are_applied_when_obstacle_circles_are_not_centered_at_origin:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 0.5], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(C := 2, 2)),
                        radii=array([0.5, 0.5], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    # Ego at origin
                    # Obstacle at (5, 0) with circles at (6, 0) and (4, 0)
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[5.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                # Closest is circle at (4, 0): ||(0,0)-(4,0)|| - 1.0 = 4 - 1 = 3.0
                expected_distances := array([[[[3.0]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Obstacle circle offsets should be applied. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_closest_across_all_obstacles_is_considered_when_multiple_obstacles_exist:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(C := 2, 2)),
                        radii=array([1.0, 0.5], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                # Obstacle A at (9, 0), Obstacle B at (4, 0)
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[9.0], [4.0]]], shape=(T, K := 2, N := 1)),
                    y=array([[[0.0], [0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0], [0.0]]], shape=(T, K, N)),
                ),
                # Obstacle B rear circle at (3,0): 3 - 1 - 0.5 = 1.5
                expected_distances := array([[[[1.5]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Should use closest circle across all obstacles. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_distances_are_computed_when_there_are_multiple_rollouts:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 1.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    # Three ego samples at different x positions
                    array(
                        [
                            [
                                [3.0, 5.0, 2.0],  # x
                                [0.0, 0.0, 0.0],  # y
                                [0.0, 0.0, 0.0],  # theta
                            ]
                        ],
                        shape=(T := 1, D_x := 3, M := 3),
                    )
                ),
                # One obstacle at origin
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[0.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [[[[1.0], [3.0], [0.0]]]], shape=(T, V := 1, M, N)
                ),  # 3-2, 5-2, 2-2
            ),
            (
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [
                            [
                                [2.0, 4.0, 1.5],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                            ]
                        ],
                        shape=(T := 1, D_x := 3, M := 3),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[0.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [[[[1.0], [3.0], [0.5]]]], shape=(T, V := 1, M, N)
                ),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Batched states should be handled correctly. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_distances_are_computed_when_there_are_multiple_time_steps:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 1.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    # Ego stationary at origin
                    array(
                        [
                            [[0.0], [0.0], [0.0]],
                            [[0.0], [0.0], [0.0]],
                            [[0.0], [0.0], [0.0]],
                        ],
                        shape=(T := 3, D_x := 3, M := 1),
                    )
                ),
                # Obstacle moving: t=0 at (5,0), t=1 at (4,0), t=2 at (3,0)
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[5.0]], [[4.0]], [[3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]], [[0.0]], [[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]], [[0.0]], [[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [[[[3.0]]], [[[2.0]]], [[[1.0]]]], shape=(T, V := 1, M, N)
                ),  # 5-2, 4-2, 3-2
            ),
            (
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [
                            [[0.0], [0.0], [0.0]],
                            [[0.0], [0.0], [0.0]],
                            [[0.0], [0.0], [0.0]],
                        ],
                        shape=(T := 3, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[4.0]], [[3.0]], [[2.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]], [[0.0]], [[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]], [[0.0]], [[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [[[[3.0]]], [[[2.0]]], [[[1.0]]]], shape=(T, V := 1, M, N)
                ),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Multiple time steps should be handled correctly. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_distance_is_computed_correctly_when_obstacle_circle_has_zero_radius:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 0.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[4.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [[[[3.0]]]], shape=(T, V := 1, M, N := 1)
                ),  # 4 - 1 - 0
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Zero radius circles should be handled. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_distance_is_infinite_when_no_obstacles_are_present:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            *[
                (
                    extractor := distance.circles(
                        ego=ego,
                        obstacle=Circles(
                            origins=np.empty((C := 0, 2)),
                            radii=np.empty((C,)),
                        ),
                        position_extractor=lambda states: types.positions(
                            x=states.array[:, 0, :],
                            y=states.array[:, 1, :],
                        ),
                        heading_extractor=lambda states: types.headings(
                            heading=states.array[:, 2, :],
                        ),
                        obstacle_position_extractor=lambda states: states.positions(),
                        obstacle_heading_extractor=lambda states: states.headings(),
                    ),
                    states := data.state_batch(
                        array(
                            [[[0.0], [0.0], [0.0]]],
                            shape=(T := 1, D_x := 3, M := 1),
                        )
                    ),
                    # No obstacles means infinite distance (represented as a large value)
                    obstacle_states := data.obstacle_2d_pose_samples(
                        x=np.empty((T, K := 0, N := 1)),
                        y=np.empty((T, K, N)),
                        heading=np.empty((T, K, N)),
                    ),
                    expected_distances,
                )
                for ego, expected_distances in (
                    (
                        Circles(
                            origins=array([[0.0, 0.0]], shape=(1, 2)),
                            radii=array([1.0], shape=(1,)),
                        ),
                        array([[[[np.inf]]]], shape=(1, 1, 1, 1)),
                    ),
                    (
                        Circles(
                            origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(2, 2)),
                            radii=array([0.5, 0.5], shape=(2,)),
                        ),
                        array([[[[np.inf]], [[np.inf]]]], shape=(1, 2, 1, 1)),
                    ),
                )
            ],
            (
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=np.empty((T, K := 0, N := 1)),
                    y=np.empty((T, K, N)),
                    heading=np.empty((T, K, N)),
                ),
                expected_distances := array(
                    [[[[np.inf]]]], shape=(T, V := 1, M, N := 1)
                ),
            ),
            (
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=np.full((T, K := 3, N := 2), np.nan),
                    y=np.full((T, K, N), np.nan),
                    heading=np.full((T, K, N), np.nan),
                ),
                expected_distances := array(
                    [[[[np.inf, np.inf]]]], shape=(T, V := 1, M, N := 2)
                ),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"No obstacles should result in infinite distance. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_distance_accounts_for_obstacle_heading:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(C := 2, 2)),
                        radii=array([0.5, 0.5], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array([[[0.0], [0.0], [0.0]]], shape=(T := 1, D_x := 3, M := 1))
                ),
                # Ego at (0,0), obstacle center at (0,3)
                # Front circle at (0,4), rear circle at (0,2) in global
                # Distance to rear circle (closest): 2 - 1 - 0.5 = 0.5
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[0.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[3.0]]], shape=(T, K, N)),
                    heading=array([[[np.pi / 2]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[0.5]]]], shape=(T, V := 1, M, N)),
            ),
            (
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(C := 2, 2)),
                        radii=array([0.5, 0.5], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array([[[0.0], [0.0], [0.0]]], shape=(T := 1, D_x := 3, M := 1))
                ),
                # rear circle at (3 - sqrt(2)/2, -sqrt(2)/2) ≈ (2.293, -0.707)
                # distance = sqrt(2.293^2 + 0.707^2) - 1.5 ≈ 2.4 - 1.5 = 0.9
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[np.pi / 4]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [
                        [
                            [
                                [
                                    np.sqrt(
                                        (3 - np.sqrt(2) / 2) ** 2
                                        + (np.sqrt(2) / 2) ** 2
                                    )
                                    - 1.5
                                ]
                            ]
                        ]
                    ],
                    shape=(T, V := 1, M, N),
                ),
            ),
            (  # Obstacle rotated 90 degrees (SAT)
                extractor := distance.sat(
                    ego=ConvexPolygon.rectangle(length=2.0, width=1.0),
                    obstacle=ConvexPolygon.rectangle(length=2.0, width=1.0),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[np.pi / 2]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[1.5]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Obstacle heading should rotate circle offsets correctly. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_distance_accounts_for_ego_heading:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (  # Ego facing up (pi/2), front circle should be above ego center
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(V := 2, 2)),
                        radii=array([0.5, 0.5], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([0.5], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [np.pi / 2]]], shape=(T := 1, D_x := 3, M := 1)
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[0.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[3.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                # Front ego circle at (0, 1), rear at (0, -1)
                # Distance from front (0,1) to obstacle (0,3): 3-1 - 0.5 - 0.5 = 1.0
                # Distance from rear (0,-1) to obstacle (0,3): 4 - 1.0 = 3.0
                expected_distances := array(
                    [[[[1.0]], [[3.0]]]], shape=(T, V := 2, M, N)
                ),
            ),
            (  # Ego facing 45 degrees (pi/4)
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(V := 2, 2)),
                        radii=array([0.5, 0.5], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([0.5], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [np.pi / 4]]], shape=(T := 1, D_x := 3, M := 1)
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [
                        [
                            # Front circle at (cos(pi/4), sin(pi/4)) = (sqrt(2)/2, sqrt(2)/2)
                            # Distance = sqrt((3 - sqrt(2)/2)^2 + (sqrt(2)/2)^2) - 1.0
                            [
                                [
                                    np.sqrt(
                                        (3 - np.sqrt(2) / 2) ** 2
                                        + (np.sqrt(2) / 2) ** 2
                                    )
                                    - 1.0
                                ]
                            ],
                            # Rear circle at (-sqrt(2)/2, -sqrt(2)/2)
                            # Distance = sqrt((3 + sqrt(2)/2)^2 + (sqrt(2)/2)^2) - 1.0
                            [
                                [
                                    np.sqrt(
                                        (3 + np.sqrt(2) / 2) ** 2
                                        + (np.sqrt(2) / 2) ** 2
                                    )
                                    - 1.0
                                ]
                            ],
                        ]
                    ],
                    shape=(T, V := 2, M, N),
                ),
            ),
            (  # Ego rotated 90 degrees (SAT)
                extractor := distance.sat(
                    ego=ConvexPolygon.rectangle(length=2.0, width=1.0),
                    obstacle=ConvexPolygon.rectangle(length=2.0, width=1.0),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [np.pi / 2]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[1.5]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Ego heading should rotate ego circle offsets correctly. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_distance_is_computed_correctly_when_multiple_samples_of_obstacle_states_are_provided:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (  # Single timestep, single ego circle, single obstacle, multiple samples
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 1.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[x := 0.0], [y := 0.0], [theta := 0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    # Obstacle sample 0 at (5, 0), sample 1 at (7, 0)
                    x=array([[[5.0, 7.0]]], shape=(T, K := 1, N := 2)),
                    y=array([[[0.0, 0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0, 0.0]]], shape=(T, K, N)),
                ),
                # Expected shape: (T, V, M, N) = (1, 1, 1, 2)
                # Sample 0: 5 - 1 - 1 = 3.0, Sample 1: 7 - 1 - 1 = 5.0
                expected_distances := array([[[[3.0, 5.0]]]], shape=(T, V := 1, M, N)),
            ),
            (  # Multiple rollouts, multiple samples
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 1.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]]],
                        shape=(T := 1, D_x := 3, M := 2),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    # Samples of one obstacle at (6, 0) and (8, 0)
                    x=array([[[6.0, 8.0]]], shape=(T, K := 1, N := 2)),
                    y=array([[[0.0, 0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0, 0.0]]], shape=(T, K, N)),
                ),
                # Rollout 0 (x=0): sample 0 -> 6-2=4, sample 1 -> 8-2=6
                # Rollout 1 (x=2): sample 0 -> 6-2-2=2, sample 1 -> 8-2-2=4
                expected_distances := array(
                    [[[[4.0, 6.0], [2.0, 4.0]]]], shape=(T, V := 1, M, N)
                ),
            ),
            (  # Multiple timesteps, Multiple ego circles, No Obstacles, Multiple samples
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0], [2.0, 0.0]], shape=(V := 2, 2)),
                        radii=array([r := 1.0, r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0], [-1.0, -1.0]], shape=(C := 2, 2)),
                        radii=array([r_o := 1.0, r_o := 1.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [
                            [[0.0], [0.0], [0.0]],
                            [[0.0], [0.0], [0.0]],
                            [[0.0], [0.0], [0.0]],
                        ],
                        shape=(T := 3, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=np.empty((T, K := 0, N := 2)),
                    y=np.empty((T, K, N)),
                    heading=np.empty((T, K, N)),
                ),
                expected_distances := np.full((T, V := 2, M, N), np.inf),
            ),
            (  # Single timestep, single obstacle, multiple samples (SAT)
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[3.0, 5.0]]], shape=(T, K := 1, N := 2)),
                    y=array([[[0.0, 0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0, 0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[2.0, 4.0]]]], shape=(T, V := 1, M, N)),
            ),
            (  # Multiple rollouts, multiple samples (SAT)
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]]],
                        shape=(T := 1, D_x := 3, M := 2),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[4.0, 6.0]]], shape=(T, K := 1, N := 2)),
                    y=array([[[0.0, 0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0, 0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [[[[3.0, 5.0], [1.0, 3.0]]]], shape=(T, V := 1, M, N)
                ),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
        ), (
            f"Distance should be computed correctly for multiple obstacle samples. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_distance_is_infinite_for_missing_obstacle_states:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.circles(
                    ego=Circles(
                        origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                        radii=array([r := 1.0], shape=(V,)),
                    ),
                    obstacle=Circles(
                        origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                        radii=array([r_o := 1.0], shape=(C,)),
                    ),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [
                            [[0.0], [0.0], [0.0]],
                            [[0.0], [0.0], [0.0]],
                            [[0.0], [0.0], [0.0]],
                        ],
                        shape=(T := 3, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    # T=0: All obstacle states missing.
                    # T=1: Some states of one sample are not missing.
                    # T=2: All states are present for one sample.
                    x=array(
                        [
                            [[np.nan, np.nan], [np.nan, np.nan]],
                            [[np.nan, 0.5], [np.nan, np.nan]],
                            [[1.0, np.nan], [np.nan, np.nan]],
                        ],
                        shape=(T, K := 2, N := 2),
                    ),
                    y=array(
                        [
                            [[np.nan, np.nan], [np.nan, np.nan]],
                            [[np.nan, np.nan], [np.nan, np.nan]],
                            [[0.0, np.nan], [np.nan, np.nan]],
                        ],
                        shape=(T, K, N),
                    ),
                    heading=array(
                        [
                            [[np.nan, np.nan], [np.nan, np.nan]],
                            [[np.nan, 0.0], [np.nan, np.nan]],
                            [[0.0, np.nan], [np.nan, np.nan]],
                        ],
                        shape=(T, K, N),
                    ),
                ),
                # Indices (T, N)
                infinite_distance_indices := [
                    # For T=0, all distances should be infinite
                    (0, 0),
                    (0, 1),
                    # For T=1, all distances should be infinite since no obstacle has all states
                    (1, 0),
                    (1, 1),
                    # For T=2, only obstacle sample 1 has all states present
                    (2, 1),
                ],
                finite_distance_indices := [(2, 0)],
            ),
            (  # SAT with missing obstacle states
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [
                            [[0.0], [0.0], [0.0]],
                            [[0.0], [0.0], [0.0]],
                            [[0.0], [0.0], [0.0]],
                        ],
                        shape=(T := 3, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array(
                        [
                            [[np.nan, np.nan], [np.nan, np.nan]],
                            [[np.nan, 5.0], [np.nan, np.nan]],
                            [[3.0, np.nan], [np.nan, np.nan]],
                        ],
                        shape=(T, K := 2, N := 2),
                    ),
                    y=array(
                        [
                            [[np.nan, np.nan], [np.nan, np.nan]],
                            [[np.nan, 0.0], [np.nan, np.nan]],
                            [[0.0, np.nan], [np.nan, np.nan]],
                        ],
                        shape=(T, K, N),
                    ),
                    heading=array(
                        [
                            [[np.nan, np.nan], [np.nan, np.nan]],
                            [[np.nan, 0.0], [np.nan, np.nan]],
                            [[0.0, np.nan], [np.nan, np.nan]],
                        ],
                        shape=(T, K, N),
                    ),
                ),
                infinite_distance_indices := [
                    (0, 0),
                    (0, 1),
                    (1, 0),
                    (2, 1),
                ],
                finite_distance_indices := [(1, 1), (2, 0)],
            ),
        ]

    @mark.parametrize(
        [
            "extractor",
            "states",
            "obstacle_states",
            "infinite_distance_indices",
            "finite_distance_indices",
        ],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        infinite_distance_indices: list[tuple[int, int]],
        finite_distance_indices: list[tuple[int, int]],
    ) -> None:
        distances = np.asarray(
            extractor(states=states, obstacle_states=obstacle_states)
        )

        assert all(
            np.isinf(distances[t, :, :, n]).all() for t, n in infinite_distance_indices
        )
        assert all(
            np.isfinite(distances[t, :, :, n]).all() for t, n in finite_distance_indices
        )


class test_that_sat_distance_is_computed_correctly_for_axis_aligned_rectangles:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (  # Separation along x-axis
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[x := 0.0], [y := 0.0], [theta := 0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[x_o := 3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[y_o := 0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[2.0]]]], shape=(T, V := 1, M, N)),
            ),
            (  # Separation along y-axis
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[x := 0.0], [y := 0.0], [theta := 0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[0.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[4.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[3.0]]]], shape=(T, V := 1, M, N)),
            ),
            (  # Diagonal separation
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[3.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[2.0]]]], shape=(T, V := 1, M, N)),
            ),
            (  # Obstacle at origin
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[3.0], [3.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[0.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[2.0]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
            atol=1e-6,
        ), (
            f"SAT distance should be computed correctly. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_sat_distance_is_computed_correctly_for_rotated_polygons:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (  # Both polygons rotated 45 degrees
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [np.pi / 4]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[np.pi / 4]]], shape=(T, K, N)),
                ),
                expected_distances := array(
                    [[[[3.0 * np.sqrt(2) / 2 - 1.0]]]],
                    shape=(T, V := 1, M, N),
                ),
            ),
            (  # Ego rotated 180 degrees (should be same as 0 for square)
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [np.pi]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[2.0]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
            atol=1e-6,
        ), (
            f"SAT with rotated polygons should compute correct distance. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_sat_distance_is_positive_when_separation_exists:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (  # Arbitrary angle (30 degrees)
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [np.pi / 6]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[5.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
    ) -> None:
        distances = extractor(states=states, obstacle_states=obstacle_states)
        assert np.all(np.isfinite(distances.array))
        assert np.all(distances.array > 0)


class test_that_sat_distance_is_zero_when_polygons_are_touching_edge_to_edge:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[1.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            0.0,
            atol=1e-6,
        ), f"Touching polygons should have distance 0. Expected: 0.0, Got: {computed}"


class test_that_sat_distance_is_negative_when_polygons_are_penetrating:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[0.5]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[-0.5]]]], shape=(T, V := 1, M, N)),
            ),
            (  # Full overlap
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[0.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[-1.0]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
            atol=1e-6,
        ), (
            f"Penetrating polygons should have negative distance. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_sat_works_with_non_rectangular_convex_polygons:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        triangle = array(
            [[0.0, 0.5], [0.5, -0.5], [-0.5, -0.5]],
            shape=(3, 2),
        )

        return [
            (
                extractor := distance.sat(
                    ego=ConvexPolygon(vertices=triangle),
                    obstacle=ConvexPolygon(vertices=triangle),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[3.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
    ) -> None:
        distances = extractor(states=states, obstacle_states=obstacle_states)
        assert distances.array.shape == (1, 1, 1, 1)
        assert np.all(np.isfinite(distances.array))
        assert np.all(distances.array > 0)


class test_that_sat_uses_closest_obstacle_when_multiple_obstacles_exist:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        return [
            (
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[10.0], [3.0]]], shape=(T, K := 2, N := 1)),
                    y=array([[[0.0], [0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0], [0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[2.0]]]], shape=(T, V := 1, M, N)),
            ),
            (  # Three obstacles at different positions
                extractor := distance.sat(
                    ego=ConvexPolygon.square(),
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[5.0], [10.0], [2.0]]], shape=(T, K := 3, N := 1)),
                    y=array([[[0.0], [0.0], [0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0], [0.0], [0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[1.0]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
            atol=1e-6,
        ), (
            f"SAT should use closest obstacle. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_sat_works_with_asymmetric_polygons:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        rectangle = ConvexPolygon.rectangle(length=4.0, width=1.0)

        return [
            (  # Rectangle ego vs square obstacle
                extractor := distance.sat(
                    ego=rectangle,
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[4.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
                expected_distances := array([[[[1.5]]]], shape=(T, V := 1, M, N)),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states", "expected_distances"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
        expected_distances: Array,
    ) -> None:
        assert np.allclose(
            computed := extractor(states=states, obstacle_states=obstacle_states),
            expected_distances,
            atol=1e-6,
        ), (
            f"SAT with asymmetric polygons should compute correct distance. "
            f"Expected: {expected_distances}, Got: {computed}"
        )


class test_that_sat_works_with_complex_convex_polygons:
    @staticmethod
    def cases(distance, types, data) -> Sequence[tuple]:
        pentagon = ConvexPolygon(
            vertices=array(
                [
                    [0.0, 1.0],
                    [0.951, 0.309],
                    [0.588, -0.809],
                    [-0.588, -0.809],
                    [-0.951, 0.309],
                ],
                shape=(5, 2),
            )
        )

        return [
            (  # Pentagon ego vs square obstacle
                extractor := distance.sat(
                    ego=pentagon,
                    obstacle=ConvexPolygon.square(),
                    position_extractor=lambda states: types.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    heading_extractor=lambda states: types.headings(
                        heading=states.array[:, 2, :],
                    ),
                    obstacle_position_extractor=lambda states: states.positions(),
                    obstacle_heading_extractor=lambda states: states.headings(),
                ),
                states := data.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T := 1, D_x := 3, M := 1),
                    )
                ),
                obstacle_states := data.obstacle_2d_pose_samples(
                    x=array([[[5.0]]], shape=(T, K := 1, N := 1)),
                    y=array([[[0.0]]], shape=(T, K, N)),
                    heading=array([[[0.0]]], shape=(T, K, N)),
                ),
            ),
        ]

    @mark.parametrize(
        ["extractor", "states", "obstacle_states"],
        [
            *cases(distance=distance.numpy, types=types.numpy, data=data.numpy),
            *cases(distance=distance.jax, types=types.jax, data=data.jax),
        ],
    )
    def test[DistanceT: Distance, ObstacleStatesT: ObstacleStates, StateT: StateBatch](
        self,
        extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
        states: StateT,
        obstacle_states: ObstacleStatesT,
    ) -> None:
        distances = extractor(states=states, obstacle_states=obstacle_states)
        assert np.all(np.isfinite(distances.array))
        assert np.all(distances.array > 0)
