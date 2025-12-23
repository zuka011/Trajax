from trajax import (
    types,
    Circles,
    Distance,
    DistanceExtractor,
    StateBatch,
    ObstacleStates,
    distance,
)

from numtypes import array, Array

import numpy as np
import jax.numpy as jnp

from tests.dsl import mppi as data, stubs
from pytest import mark


@mark.parametrize(
    ["extractor", "states", "expected_distances"],
    [
        # Basic distance along x-axis
        (
            extractor := distance.numpy.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(  # K = 1 Obstacle
                        x=array([[x_o := 5.0]], shape=(T := 1, K := 1)),
                        y=array([[y_o := 0.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                array(
                    [[[x := 0.0], [y := 0.0], [theta := 0.0]]],
                    shape=(T, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[x_o - x - r - r_o]]], shape=(T, V := 1, M)),
        ),
        # Distance with diagonal (3-4-5 triangle)
        (
            extractor := distance.numpy.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        x=array([[x_o := 3.0]], shape=(T := 1, K := 1)),
                        y=array([[y_o := 4.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                array(
                    [[[x := 0.0], [y := 0.0], [theta := 0.0]]],
                    shape=(T, D_x := 3, M := 1),
                )
            ),
            # distance = sqrt(3^2 + 4^2) - 1 - 1 = 5 - 2 = 3
            expected_distances := array([[[3.0]]], shape=(T, V := 1, M)),
        ),
        # Penetration (negative distance)
        (
            extractor := distance.numpy.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        x=array([[x_o := 1.0]], shape=(T := 1, K := 1)),
                        y=array([[y_o := 0.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                array(
                    [[[x := 0.0], [y := 0.0], [theta := 0.0]]],
                    shape=(T, D_x := 3, M := 1),
                )
            ),
            # distance = 1 - 1 - 1 = -1 (penetration)
            expected_distances := array([[[-1.0]]], shape=(T, V := 1, M)),
        ),
        (  # Equivalent JAX tests
            extractor := distance.jax.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(  # K = 1 Obstacle
                        x=jnp.array([[x_o := 5.0]]),
                        y=jnp.array([[y_o := 0.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array(
                    [[[x := 0.0], [y := 0.0], [theta := 0.0]]],
                    shape=(T := 1, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[x_o - x - r - r_o]]], shape=(T, V := 1, M)),
        ),
        (
            extractor := distance.jax.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.array([[3.0]]),
                        y=jnp.array([[4.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T := 1, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[3.0]]], shape=(T, V := 1, M)),
        ),
        (
            extractor := distance.jax.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.array([[1.0]]),
                        y=jnp.array([[0.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T := 1, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[-1.0]]], shape=(T, V := 1, M)),
        ),
    ],
)
def test_that_distance_is_computed_correctly_when_ego_and_obstacle_are_single_circles[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    expected_distances: Array,
) -> None:
    assert np.allclose(distance := extractor(states), expected_distances), (
        f"Distance should be computed correctly. "
        f"Expected: {expected_distances}, Got: {distance}"
    )


@mark.parametrize(
    ["extractor", "states", "expected_distances"],
    [
        (
            extractor := distance.numpy.circles(
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
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        x=array([[10.0]], shape=(T := 1, K := 1)),
                        y=array([[0.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[8.5], [6.5], [4.5]]], shape=(T, V := 3, M)),
        ),
        (
            extractor := distance.jax.circles(
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
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.array([[10.0]]),
                        y=jnp.array([[0.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T := 1, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[8.5], [6.5], [4.5]]], shape=(T, V := 3, M)),
        ),
    ],
)
def test_that_that_distance_is_computed_correctly_when_ego_is_multiple_circles[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    expected_distances: Array,
) -> None:
    assert np.allclose(distance := extractor(states), expected_distances), (
        f"Each ego part should have its own distance. "
        f"Expected: {expected_distances}, Got: {distance}"
    )


@mark.parametrize(
    ["extractor", "states", "expected_distances"],
    [
        (
            extractor := distance.numpy.circles(
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
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        x=array([[0.0]], shape=(T := 1, K := 1)),
                        y=array([[0.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T, D_x := 3, M := 1),
                )
            ),
            # Closest is circle at (5,0) with r=0.5: 5 - 1 - 0.5 = 3.5
            expected_distances := array([[[3.5]]], shape=(T, V := 1, M)),
        ),
        (
            extractor := distance.jax.circles(
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
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.array([[0.0]]),
                        y=jnp.array([[0.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T := 1, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[3.5]]], shape=(T, V := 1, M)),
        ),
    ],
)
def test_that_closest_obstacle_circle_is_used_when_obstacle_is_multiple_circles[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    expected_distances: Array,
) -> None:
    assert np.allclose(distance := extractor(states), expected_distances), (
        f"Should use closest obstacle circle. "
        f"Expected: {expected_distances}, Got: {distance}"
    )


@mark.parametrize(
    ["extractor", "states", "expected_distances"],
    [
        # Ego has two circles offset along x-axis
        (
            extractor := distance.numpy.circles(
                ego=Circles(
                    origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(V := 2, 2)),
                    radii=array([0.5, 0.5], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 0.5], shape=(C,)),
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        x=array([[5.0]], shape=(T := 1, K := 1)),
                        y=array([[0.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                # Ego center at (0, 0)
                # Circle 1 at (1, 0), Circle 2 at (-1, 0)
                # Obstacle at (5, 0)
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T, D_x := 3, M := 1),
                )
            ),
            expected_distances := array(
                [
                    [
                        [3.0],  # ||(5,0)-(1,0)|| - 1.0 = 4 - 1 = 3.0
                        [5.0],  # ||(5,0)-(-1,0)|| - 1.0 = 6 - 1 = 5.0
                    ]
                ],
                shape=(T, V := 2, M),
            ),
        ),
        (
            extractor := distance.jax.circles(
                ego=Circles(
                    origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(V := 2, 2)),
                    radii=array([0.5, 0.5], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 0.5], shape=(C,)),
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.array([[5.0]]),
                        y=jnp.array([[0.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array([[[0.0], [0.0], [0.0]]], shape=(T := 1, D_x := 3, M := 1))
            ),
            expected_distances := array([[[3.0], [5.0]]], shape=(T, V := 2, M := 1)),
        ),
    ],
)
def test_that_ego_circle_offsets_are_applied_when_ego_circles_are_not_centered_at_origin[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    expected_distances: Array,
) -> None:
    assert np.allclose(distance := extractor(states), expected_distances), (
        f"Ego circle offsets should be applied. "
        f"Expected: {expected_distances}, Got: {distance}"
    )


@mark.parametrize(
    ["extractor", "states", "expected_distances"],
    [
        (
            extractor := distance.numpy.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 0.5], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(C := 2, 2)),
                    radii=array([0.5, 0.5], shape=(C,)),
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        x=array([[5.0]], shape=(T := 1, K := 1)),
                        y=array([[0.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                # Ego at origin
                # Obstacle at (5, 0) with circles at (6, 0) and (4, 0)
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T, D_x := 3, M := 1),
                )
            ),
            # Closest is circle at (4, 0): ||(0,0)-(4,0)|| - 1.0 = 4 - 1 = 3.0
            expected_distances := array([[[3.0]]], shape=(T, V := 1, M)),
        ),
        (
            extractor := distance.jax.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 0.5], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(C := 2, 2)),
                    radii=array([0.5, 0.5], shape=(C,)),
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.array([[5.0]]),
                        y=jnp.array([[0.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T := 1, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[3.0]]], shape=(T, V := 1, M)),
        ),
    ],
)
def test_that_obstacle_circle_offsets_are_applied_when_obstacle_circles_are_not_centered_at_origin[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    expected_distances: Array,
) -> None:
    assert np.allclose(distance := extractor(states), expected_distances), (
        f"Obstacle circle offsets should be applied. "
        f"Expected: {expected_distances}, Got: {distance}"
    )


@mark.parametrize(
    ["extractor", "states", "expected_distances"],
    [
        (
            extractor := distance.numpy.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(C := 2, 2)),
                    radii=array([1.0, 0.5], shape=(C,)),
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        # Obstacle A at (9, 0), Obstacle B at (4, 0)
                        x=array([[9.0, 4.0]], shape=(T := 1, K := 2)),
                        y=array([[0.0, 0.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T, D_x := 3, M := 1),
                )
            ),
            # Obstacle B rear circle at (3,0): 3 - 1 - 0.5 = 1.5
            expected_distances := array([[[1.5]]], shape=(T, V := 1, M)),
        ),
        (
            extractor := distance.jax.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(C := 2, 2)),
                    radii=array([1.0, 0.5], shape=(C,)),
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.array([[9.0, 4.0]]),
                        y=jnp.array([[0.0, 0.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T := 1, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[1.5]]], shape=(T, V := 1, M)),
        ),
    ],
)
def test_that_closest_across_all_obstacles_is_considered_when_multiple_obstacles_exist[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    expected_distances: Array,
) -> None:
    assert np.allclose(distance := extractor(states), expected_distances), (
        f"Should use closest circle across all obstacles. "
        f"Expected: {expected_distances}, Got: {distance}"
    )


@mark.parametrize(
    ["extractor", "states", "expected_distances"],
    [
        (
            extractor := distance.numpy.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        # One obstacle at origin
                        x=array([[0.0]], shape=(T := 1, K := 1)),
                        y=array([[0.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                # Three ego samples at different x positions
                array(
                    [
                        [
                            [3.0, 5.0, 2.0],  # x
                            [0.0, 0.0, 0.0],  # y
                            [0.0, 0.0, 0.0],  # theta
                        ]
                    ],
                    shape=(T, D_x := 3, M := 3),
                )
            ),
            expected_distances := array(
                [[[1.0, 3.0, 0.0]]], shape=(T, V := 1, M)
            ),  # 3-2, 5-2, 2-2
        ),
        (
            extractor := distance.jax.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.array([[0.0]]),
                        y=jnp.array([[0.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array(
                    [
                        [
                            [3.0, 5.0, 2.0],
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ]
                    ],
                    shape=(T := 1, D_x := 3, M := 3),
                )
            ),
            expected_distances := array([[[1.0, 3.0, 0.0]]], shape=(T, V := 1, M)),
        ),
    ],
)
def test_that_distances_are_computed_when_there_are_multiple_rollouts[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    expected_distances: Array,
) -> None:
    assert np.allclose(distance := extractor(states), expected_distances), (
        f"Batched states should be handled correctly. "
        f"Expected: {expected_distances}, Got: {distance}"
    )


@mark.parametrize(
    ["extractor", "states", "expected_distances"],
    [
        (
            extractor := distance.numpy.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        # Obstacle moving: t=0 at (5,0), t=1 at (4,0), t=2 at (3,0)
                        x=array([[5.0], [4.0], [3.0]], shape=(T := 3, K := 1)),
                        y=array([[0.0], [0.0], [0.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                # Ego stationary at origin
                array(
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    shape=(T, D_x := 3, M := 1),
                )
            ),
            expected_distances := array(
                [[[3.0]], [[2.0]], [[1.0]]], shape=(T, V := 1, M)
            ),  # 5-2, 4-2, 3-2
        ),
        (
            extractor := distance.jax.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.array([[5.0], [4.0], [3.0]]),
                        y=jnp.array([[0.0], [0.0], [0.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array(
                    [
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                        [[0.0], [0.0], [0.0]],
                    ],
                    shape=(T := 3, D_x := 3, M := 1),
                )
            ),
            expected_distances := array(
                [[[3.0]], [[2.0]], [[1.0]]], shape=(T, V := 1, M)
            ),
        ),
    ],
)
def test_that_distances_are_computed_when_there_are_multiple_time_steps[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    expected_distances: Array,
) -> None:
    assert np.allclose(distance := extractor(states), expected_distances), (
        f"Multiple time steps should be handled correctly. "
        f"Expected: {expected_distances}, Got: {distance}"
    )


@mark.parametrize(
    ["extractor", "states", "expected_distances"],
    [
        (
            extractor := distance.numpy.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 0.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        x=array([[4.0]], shape=(T := 1, K := 1)),
                        y=array([[0.0]], shape=(T, K)),
                    )
                ),
            ),
            states := data.numpy.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[3.0]]], shape=(T, V := 1, M)),  # 4 - 1 - 0
        ),
        (
            extractor := distance.jax.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 0.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.array([[4.0]]),
                        y=jnp.array([[0.0]]),
                    )
                ),
            ),
            states := data.jax.state_batch(
                array(
                    [[[0.0], [0.0], [0.0]]],
                    shape=(T := 1, D_x := 3, M := 1),
                )
            ),
            expected_distances := array([[[3.0]]], shape=(T, V := 1, M)),
        ),
    ],
)
def test_that_distance_is_computed_correctly_when_obstacle_circle_has_zero_radius[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    expected_distances: Array,
) -> None:
    assert np.allclose(distance := extractor(states), expected_distances), (
        f"Zero radius circles should be handled. "
        f"Expected: {expected_distances}, Got: {distance}"
    )


@mark.parametrize(
    ["extractor", "states", "expected_distances"],
    [
        *[
            (
                extractor := distance.numpy.circles(
                    ego=ego,
                    obstacle=Circles(
                        origins=np.empty((C := 0, 2)),
                        radii=np.empty((C,)),
                    ),
                    position_extractor=lambda states: types.numpy.positions(
                        x=np.asarray(states)[:, 0, :],
                        y=np.asarray(states)[:, 1, :],
                    ),
                    obstacle_states=stubs.ObstacleStateProvider.returns(
                        types.numpy.obstacle_states.create(
                            x=np.empty((T := 1, K := 0)),
                            y=np.empty((T, K)),
                        )
                    ),
                ),
                states := data.numpy.state_batch(
                    array(
                        [[[0.0], [0.0], [0.0]]],
                        shape=(T, D_x := 3, M := 1),
                    )
                ),
                # No obstacles means infinite distance (represented as a large value)
                expected_distances,
            )
            for ego, expected_distances in (
                (
                    Circles(
                        origins=array([[0.0, 0.0]], shape=(1, 2)),
                        radii=array([1.0], shape=(1,)),
                    ),
                    array([[[np.inf]]], shape=(1, 1, 1)),
                ),
                (
                    Circles(
                        origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(2, 2)),
                        radii=array([0.5, 0.5], shape=(2,)),
                    ),
                    array([[[np.inf], [np.inf]]], shape=(1, 2, 1)),
                ),
            )
        ],
        *[
            (
                extractor := distance.jax.circles(
                    ego=ego,
                    obstacle=Circles(
                        origins=np.empty((C := 0, 2)),
                        radii=np.empty((C,)),
                    ),
                    position_extractor=lambda states: types.jax.positions(
                        x=states.array[:, 0, :],
                        y=states.array[:, 1, :],
                    ),
                    obstacle_states=stubs.ObstacleStateProvider.returns(
                        types.jax.obstacle_states.create(
                            x=jnp.empty((T := 1, K := 0)),
                            y=jnp.empty((T, K)),
                        )
                    ),
                ),
                state := data.jax.state_batch(
                    array([[[0.0], [0.0], [0.0]]], shape=(T := 1, D_x := 3, M := 1))
                ),
                expected_distances,
            )
            for ego, expected_distances in (
                (
                    Circles(
                        origins=array([[0.0, 0.0]], shape=(1, 2)),
                        radii=array([1.0], shape=(1,)),
                    ),
                    array([[[np.inf]]], shape=(1, 1, 1)),
                ),
                (
                    Circles(
                        origins=array([[1.0, 0.0], [-1.0, 0.0]], shape=(2, 2)),
                        radii=array([0.5, 0.5], shape=(2,)),
                    ),
                    array([[[np.inf], [np.inf]]], shape=(1, 2, 1)),
                ),
            )
        ],
    ],
)
def test_that_distance_is_infinite_when_no_obstacles_are_present[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    expected_distances: Array,
) -> None:
    assert np.allclose(distance := extractor(states), expected_distances), (
        f"No obstacles should result in infinite distance. "
        f"Expected: {expected_distances}, Got: {distance}"
    )


@mark.parametrize(
    ["extractor", "states", "obstacle_states", "expected_distances"],
    [
        (
            extractor := distance.numpy.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.numpy.positions(
                    x=np.asarray(states)[:, 0, :],
                    y=np.asarray(states)[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.numpy.obstacle_states.create(
                        x=np.zeros((1, 1)), y=np.zeros((1, 1))
                    )
                ),
            ),
            states := data.numpy.state_batch(
                array([[[0.0], [0.0], [0.0]]], shape=(T := 1, D_x := 3, M := 1))
            ),
            obstacle_states := types.numpy.obstacle_states.create(
                x=array([[x_o := 5.0]], shape=(T, K := 1)),
                y=array([[y_o := 0.0]], shape=(T, K)),
            ),
            expected_distances := array([[[x_o - r - r_o]]], shape=(T, V := 1, M)),
        ),
        (
            extractor := distance.jax.circles(
                ego=Circles(
                    origins=array([[0.0, 0.0]], shape=(V := 1, 2)),
                    radii=array([r := 1.0], shape=(V,)),
                ),
                obstacle=Circles(
                    origins=array([[0.0, 0.0]], shape=(C := 1, 2)),
                    radii=array([r_o := 1.0], shape=(C,)),
                ),
                position_extractor=lambda states: types.jax.positions(
                    x=states.array[:, 0, :],
                    y=states.array[:, 1, :],
                ),
                obstacle_states=stubs.ObstacleStateProvider.returns(
                    types.jax.obstacle_states.create(
                        x=jnp.zeros((1, 1)), y=jnp.zeros((1, 1))
                    )
                ),
            ),
            states := data.jax.state_batch(
                array([[[0.0], [0.0], [0.0]]], shape=(T := 1, D_x := 3, M := 1))
            ),
            obstacle_states := types.jax.obstacle_states.create(
                x=jnp.array([[x_o := 5.0]]),
                y=jnp.array([[y_o := 0.0]]),
            ),
            expected_distances := array([[[x_o - r - r_o]]], shape=(T, V := 1, M)),
        ),
    ],
)
def test_that_distance_is_computed_correctly_when_explicit_obstacle_states_are_provided[
    DistanceT: Distance,
    ObstacleStatesT: ObstacleStates,
    StateT: StateBatch,
](
    extractor: DistanceExtractor[StateT, ObstacleStatesT, DistanceT],
    states: StateT,
    obstacle_states: ObstacleStatesT,
    expected_distances: Array,
) -> None:
    assert np.allclose(
        computed := extractor.measure(states, obstacle_states), expected_distances
    ), (
        f"The distance should be measured based on the provided obstacle states. "
        f"Expected: {expected_distances}, Got: {computed}"
    )
