from typing import Sequence

from trajax import (
    CovarianceSequences,
    CovariancePropagator,
    propagator as create_propagator,
)

from numtypes import array

import numpy as np

from tests.dsl import mppi as data
from pytest import mark


class test_that_covariance_is_propagated:
    @staticmethod
    def cases(create_propagator, data) -> Sequence[tuple]:
        return [
            (  # No velocity uncertainty -> position covariance remains constant
                propagator := create_propagator.linear(
                    time_step_size=1.0,
                    initial_covariance=create_propagator.covariance.constant_variance(
                        position_variance=(var := 0.1), velocity_variance=0.0
                    ),
                ),
                states := data.obstacle_states(
                    x=array(
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], shape=(T := 3, K := 2)
                    ),
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
                propagator := create_propagator.linear(
                    time_step_size=2.0,
                    initial_covariance=create_propagator.covariance.constant_variance(
                        position_variance=3.0, velocity_variance=1.5
                    ),
                ),
                states := data.obstacle_states(
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
                propagator := create_propagator.linear(
                    time_step_size=(dt := 1.0),
                    initial_covariance=create_propagator.covariance.constant_covariance(
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
                states := data.obstacle_states(
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
                propagator := create_propagator.linear(
                    time_step_size=(dt := 0.5),
                    initial_covariance=create_propagator.covariance.constant_variance(
                        position_variance=(p := 0.0), velocity_variance=(v := 2.0)
                    ),
                ),
                states := data.obstacle_states(
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
                propagator := create_propagator.linear(
                    time_step_size=1.0,
                    initial_covariance=create_propagator.covariance.constant_variance(
                        position_variance=0.1, velocity_variance=0.5
                    ),
                ),
                states := data.obstacle_states(
                    x=np.empty((T := 3, K := 0)),
                    y=np.empty((T, K)),
                ),
                expected := np.empty((T, 2, 2, K)),
            ),
        ]

    @mark.parametrize(
        ["propagator", "states", "expected"],
        [
            *cases(create_propagator=create_propagator.numpy, data=data.numpy),
            *cases(create_propagator=create_propagator.jax, data=data.jax),
        ],
    )
    def test[StateSequencesT, CovarianceSequencesT: CovarianceSequences](
        self,
        propagator: CovariancePropagator[StateSequencesT, CovarianceSequencesT],
        states: StateSequencesT,
        expected: CovarianceSequencesT,
    ) -> None:
        assert np.allclose(propagator.propagate(states=states), expected)


class test_that_covariance_is_padded_to_the_given_dimension:
    @staticmethod
    def cases(create_propagator, data) -> Sequence[tuple]:
        return [
            (
                propagator := create_propagator.linear(
                    time_step_size=1.0,
                    initial_covariance=create_propagator.covariance.constant_variance(
                        position_variance=(var := 0.1), velocity_variance=0.0
                    ),
                    # A covariance array must not be singular in any dimension.
                    padding=create_propagator.padding(
                        to_dimension=(D_p := 4), epsilon=(eps := 0.001)
                    ),
                ),
                states := data.obstacle_states(
                    x=array(
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], shape=(T := 3, K := 2)
                    ),
                    y=array([[6.0, 1.0], [4.0, 3.0], [2.0, 5.0]], shape=(T, K)),
                ),
                expected := array(
                    [
                        (
                            initial_covariance := [
                                [[var, var], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [var, var], [0.0, 0.0], [0.0, 0.0]],
                                [[0.0, 0.0], [0.0, 0.0], [eps, eps], [0.0, 0.0]],
                                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [eps, eps]],
                            ]
                        ),
                        initial_covariance,
                        initial_covariance,
                    ],
                    shape=(T, D_p, D_p, K),
                ),
            ),
            (  # No padding is needed if the dimension is already sufficient.
                propagator := create_propagator.linear(
                    time_step_size=1.0,
                    initial_covariance=create_propagator.covariance.constant_variance(
                        position_variance=(var := 0.1), velocity_variance=0.0
                    ),
                    padding=create_propagator.padding(
                        to_dimension=(D_p := 2), epsilon=(eps := 0.001)
                    ),
                ),
                states := data.obstacle_states(
                    x=array(
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], shape=(T := 3, K := 2)
                    ),
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
                    shape=(T, D_p, D_p, K),
                ),
            ),
        ]

    @mark.parametrize(
        ["propagator", "states", "expected"],
        [
            *cases(create_propagator=create_propagator.numpy, data=data.numpy),
            *cases(create_propagator=create_propagator.jax, data=data.jax),
        ],
    )
    def test[StateSequencesT, CovarianceSequencesT: CovarianceSequences](
        self,
        propagator: CovariancePropagator[StateSequencesT, CovarianceSequencesT],
        states: StateSequencesT,
        expected: CovarianceSequencesT,
    ) -> None:
        assert np.allclose(propagator.propagate(states=states), expected)
