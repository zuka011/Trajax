from dataclasses import dataclass

from trajax import (
    CovarianceSequences,
    CovariancePropagator,
    propagator as create_propagator,
    types,
)

from jaxtyping import Array as JaxArray, Float
from numtypes import array

import numpy as np
import jax.numpy as jnp

from tests.dsl import mppi as data
from pytest import mark

type NumPyObstacleStates[T: int, K: int] = types.numpy.ObstacleStates[T, K]
type NumPyInitialPositionCovariance[T: int, K: int] = (
    types.numpy.InitialPositionCovariance[K]
)
type NumPyInitialVelocityVariance[K: int] = types.numpy.InitialVelocityCovariance[K]
type NumPyInitialCovarianceProvider = types.numpy.InitialCovarianceProvider[
    NumPyObstacleStates
]

type JaxObstacleStates[T: int, K: int] = types.jax.ObstacleStates[T, K]
type JaxInitialPositionCovariance[K: int] = types.jax.InitialPositionCovariance[K]
type JaxInitialVelocityCovariance[K: int] = types.jax.InitialVelocityCovariance[K]
type JaxInitialCovarianceProvider = types.jax.InitialCovarianceProvider[
    JaxObstacleStates
]


@dataclass(kw_only=True, frozen=True)
class NumPyConstantVarianceProvider:
    position_variance: float
    velocity_variance: float

    def position[K: int](
        self, states: NumPyObstacleStates[int, K]
    ) -> NumPyInitialPositionCovariance[K]:
        return np.tile(
            (np.eye(2) * self.position_variance)[..., np.newaxis], (1, 1, states.count)
        )

    def velocity[K: int](
        self, states: NumPyObstacleStates[int, K]
    ) -> NumPyInitialVelocityVariance[K]:
        return np.tile(
            (np.eye(2) * self.velocity_variance)[..., np.newaxis], (1, 1, states.count)
        )


@dataclass(kw_only=True, frozen=True)
class NumPyConstantCovarianceProvider[K: int]:
    position_covariance: NumPyInitialPositionCovariance[K]
    velocity_covariance: NumPyInitialVelocityVariance[K]

    def position(
        self, states: NumPyObstacleStates[int, K]
    ) -> NumPyInitialPositionCovariance[K]:
        assert states.count == self.position_covariance.shape[2]
        return self.position_covariance

    def velocity(
        self, states: NumPyObstacleStates[int, K]
    ) -> NumPyInitialVelocityVariance[K]:
        assert states.count == self.velocity_covariance.shape[2]
        return self.velocity_covariance


@dataclass(kw_only=True, frozen=True)
class JaxConstantVarianceProvider:
    position_variance: float
    velocity_variance: float

    def position[K: int](
        self, states: JaxObstacleStates[int, K]
    ) -> JaxInitialPositionCovariance[K]:
        return jnp.tile(
            (jnp.eye(2) * self.position_variance)[..., jnp.newaxis],
            (1, 1, states.count),
        )

    def velocity[K: int](
        self, states: JaxObstacleStates[int, K]
    ) -> JaxInitialVelocityCovariance[K]:
        return jnp.tile(
            (jnp.eye(2) * self.velocity_variance)[..., jnp.newaxis],
            (1, 1, states.count),
        )


@dataclass(kw_only=True, frozen=True)
class JaxConstantCovarianceProvider[K: int]:
    position_covariance: Float[JaxArray, "2 2 K"]
    velocity_covariance: Float[JaxArray, "2 2 K"]

    def position(
        self, states: JaxObstacleStates[int, K]
    ) -> JaxInitialPositionCovariance[K]:
        assert states.count == self.position_covariance.shape[2]
        return self.position_covariance

    def velocity(
        self, states: JaxObstacleStates[int, K]
    ) -> JaxInitialVelocityCovariance[K]:
        assert states.count == self.velocity_covariance.shape[2]
        return self.velocity_covariance


@mark.parametrize(
    ["propagator", "states", "expected"],
    [
        (  # No velocity uncertainty -> position covariance remains constant
            propagator := create_propagator.numpy.linear(
                time_step_size=1.0,
                initial_covariance=NumPyConstantVarianceProvider(
                    position_variance=(var := 0.1), velocity_variance=0.0
                ),
            ),
            states := data.numpy.obstacle_states(
                x=array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], shape=(T := 3, K := 2)),
                y=array([[6.0, 1.0], [4.0, 3.0], [2.0, 5.0]], shape=(T, K)),
            ),
            expected := array(
                [
                    (
                        initial_covariance := [
                            [[var, var], [0.0, 0.0]],
                            [[0.0, 0.0], [var, var]],
                        ]
                    ),
                    initial_covariance,
                    initial_covariance,
                ],
                shape=(T, D_p := 2, D_p, K),
            ),
        ),
        (  # Linear growth: t * dt² * v_var
            propagator := create_propagator.numpy.linear(
                time_step_size=2.0,
                initial_covariance=NumPyConstantVarianceProvider(
                    position_variance=3.0, velocity_variance=1.5
                ),
            ),
            states := data.numpy.obstacle_states(
                x=array([[0.0], [1.0], [2.0]], shape=(T := 3, K := 1)),
                y=array([[0.0], [2.0], [0.0]], shape=(T, K)),
            ),
            expected := array(
                [
                    [  # T=1
                        [[9.0], [0.0]],
                        [[0.0], [9.0]],
                    ],
                    [  # T=2
                        [[15.0], [0.0]],
                        [[0.0], [15.0]],
                    ],
                    [  # T=3
                        [[21.0], [0.0]],
                        [[0.0], [21.0]],
                    ],
                ],
                shape=(T, 2, 2, K),
            ),
        ),
        (  # Multiple obstacles with correlation.
            propagator := create_propagator.numpy.linear(
                time_step_size=(dt := 1.0),
                initial_covariance=NumPyConstantCovarianceProvider(
                    position_covariance=array(
                        [
                            [[x0 := 0.7, x1 := 0.4], [xy0 := 0.5, xy1 := 0.2]],
                            [[xy0, xy1], [y0 := 0.6, y1 := 0.2]],
                        ],
                        shape=(2, 2, K := 2),
                    ),
                    velocity_covariance=array(
                        [
                            [[vx0 := 1.0, vx1 := 2.0], [vxy0 := 0.2, vxy1 := 0.5]],
                            [[vxy0, vxy1], [vy0 := 1.5, vy1 := 2.5]],
                        ],
                        shape=(2, 2, K),
                    ),
                ),
            ),
            states := data.numpy.obstacle_states(
                x=array([[0.0, 5.0], [1.0, 6.0]], shape=(T := 2, K)),
                y=array([[0.0, 0.0], [1.0, 1.0]], shape=(T, K)),
            ),
            expected := array(
                [
                    [  # T=1: p + 1 * dt² * v
                        [
                            [(x0 + vx0), (x1 + vx1)],
                            [(xy0 + vxy0), (xy1 + vxy1)],
                        ],
                        [
                            [(xy0 + vxy0), (xy1 + vxy1)],
                            [(y0 + vy0), (y1 + vy1)],
                        ],
                    ],
                    [  # T=2: p + 2 * dt² * v
                        [
                            [(x0 + 2 * vx0), (x1 + 2 * vx1)],
                            [(xy0 + 2 * vxy0), (xy1 + 2 * vxy1)],
                        ],
                        [
                            [(xy0 + 2 * vxy0), (xy1 + 2 * vxy1)],
                            [(y0 + 2 * vy0), (y1 + 2 * vy1)],
                        ],
                    ],
                ],
                shape=(T, 2, 2, K),
            ),
        ),
        (  # Single time step
            propagator := create_propagator.numpy.linear(
                time_step_size=(dt := 0.5),
                initial_covariance=NumPyConstantVarianceProvider(
                    position_variance=(p := 0.0), velocity_variance=(v := 2.0)
                ),
            ),
            states := data.numpy.obstacle_states(
                x=array([[0.0]], shape=(T := 1, K := 1)),
                y=array([[0.0]], shape=(T, K)),
            ),
            expected := array(
                [
                    [
                        [[(expected := p + dt**2 * v)], [0.0]],
                        [[0.0], [expected]],
                    ],
                ],
                shape=(T, 2, 2, K),
            ),
        ),
        (  # No obstacles
            propagator := create_propagator.numpy.linear(
                time_step_size=1.0,
                initial_covariance=NumPyConstantVarianceProvider(
                    position_variance=0.1, velocity_variance=0.5
                ),
            ),
            states := data.numpy.obstacle_states(
                x=np.empty((T := 3, K := 0)),
                y=np.empty((T, K)),
            ),
            expected := np.empty((T, 2, 2, K)),
        ),
    ]
    + [
        (  # Analogous tests for JAX implementation
            propagator := create_propagator.jax.linear(
                time_step_size=1.0,
                initial_covariance=JaxConstantVarianceProvider(
                    position_variance=(var := 0.1), velocity_variance=0.0
                ),
            ),
            states := data.jax.obstacle_states(
                x=array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], shape=(T := 3, K := 2)),
                y=array([[6.0, 1.0], [4.0, 3.0], [2.0, 5.0]], shape=(T, K)),
            ),
            expected := array(
                [
                    (
                        initial_covariance := [
                            [[var, var], [0.0, 0.0]],
                            [[0.0, 0.0], [var, var]],
                        ]
                    ),
                    initial_covariance,
                    initial_covariance,
                ],
                shape=(T, D_p := 2, D_p, K),
            ),
        ),
        (
            propagator := create_propagator.jax.linear(
                time_step_size=2.0,
                initial_covariance=JaxConstantVarianceProvider(
                    position_variance=3.0, velocity_variance=1.5
                ),
            ),
            states := data.jax.obstacle_states(
                x=array([[0.0], [1.0], [2.0]], shape=(T := 3, K := 1)),
                y=array([[0.0], [2.0], [0.0]], shape=(T, K)),
            ),
            expected := array(
                [
                    [
                        [[9.0], [0.0]],
                        [[0.0], [9.0]],
                    ],
                    [
                        [[15.0], [0.0]],
                        [[0.0], [15.0]],
                    ],
                    [
                        [[21.0], [0.0]],
                        [[0.0], [21.0]],
                    ],
                ],
                shape=(T, 2, 2, K),
            ),
        ),
        (
            propagator := create_propagator.jax.linear(
                time_step_size=(dt := 1.0),
                initial_covariance=JaxConstantCovarianceProvider(
                    position_covariance=jnp.array(
                        [
                            [[x0 := 0.7, x1 := 0.4], [xy0 := 0.5, xy1 := 0.2]],
                            [[xy0, xy1], [y0 := 0.6, y1 := 0.2]],
                        ],
                    ),
                    velocity_covariance=jnp.array(
                        [
                            [[vx0 := 1.0, vx1 := 2.0], [vxy0 := 0.2, vxy1 := 0.5]],
                            [[vxy0, vxy1], [vy0 := 1.5, vy1 := 2.5]],
                        ],
                    ),
                ),
            ),
            states := data.jax.obstacle_states(
                x=array([[0.0, 5.0], [1.0, 6.0]], shape=(T := 2, K := 2)),
                y=array([[0.0, 0.0], [1.0, 1.0]], shape=(T, K)),
            ),
            expected := array(
                [
                    [
                        [
                            [(x0 + vx0), (x1 + vx1)],
                            [(xy0 + vxy0), (xy1 + vxy1)],
                        ],
                        [
                            [(xy0 + vxy0), (xy1 + vxy1)],
                            [(y0 + vy0), (y1 + vy1)],
                        ],
                    ],
                    [
                        [
                            [(x0 + 2 * vx0), (x1 + 2 * vx1)],
                            [(xy0 + 2 * vxy0), (xy1 + 2 * vxy1)],
                        ],
                        [
                            [(xy0 + 2 * vxy0), (xy1 + 2 * vxy1)],
                            [(y0 + 2 * vy0), (y1 + 2 * vy1)],
                        ],
                    ],
                ],
                shape=(T, 2, 2, K),
            ),
        ),
        (
            propagator := create_propagator.jax.linear(
                time_step_size=(dt := 0.5),
                initial_covariance=JaxConstantVarianceProvider(
                    position_variance=(p := 0.0), velocity_variance=(v := 2.0)
                ),
            ),
            states := data.jax.obstacle_states(
                x=array([[0.0]], shape=(T := 1, K := 1)),
                y=array([[0.0]], shape=(T, K)),
            ),
            expected := array(
                [
                    [
                        [[(expected := p + dt**2 * v)], [0.0]],
                        [[0.0], [expected]],
                    ],
                ],
                shape=(T, 2, 2, K),
            ),
        ),
        (
            propagator := create_propagator.jax.linear(
                time_step_size=1.0,
                initial_covariance=JaxConstantVarianceProvider(
                    position_variance=0.1, velocity_variance=0.5
                ),
            ),
            states := data.jax.obstacle_states(
                x=np.empty((T := 3, K := 0)),
                y=np.empty((T, K)),
            ),
            expected := np.empty((T, 2, 2, K)),
        ),
    ],
)
def test_that_covariance_is_propagated[
    StateSequencesT,
    CovarianceSequencesT: CovarianceSequences,
](
    propagator: CovariancePropagator[StateSequencesT, CovarianceSequencesT],
    states: StateSequencesT,
    expected: CovarianceSequencesT,
) -> None:
    assert np.allclose(propagator.propagate(states=states), expected)
