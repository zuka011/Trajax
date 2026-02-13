from typing import Sequence

from trajax import (
    ObstacleStates,
    ObstacleMotionPredictor,
    model,
    propagator as create_propagator,
    predictor as create_predictor,
)

from numtypes import array
from jaxtyping import ArrayLike

import numpy as np

from tests.dsl import mppi as data, prediction_creator, check
from pytest import mark


class ExtractXCoordinateCovariance:
    """Extracts only the X-coordinate covariance (1x1) from the full covariance matrix.

    Not a realistic covariance extractor, but useful for testing.
    """

    def __call__(self, covariance: ArrayLike) -> ArrayLike:
        return covariance[:, :1, :1, :]


class test_that_covariance_is_resized:
    @staticmethod
    def cases(
        create_propagator, create_predictor, data, model, prediction_creator
    ) -> Sequence[tuple]:

        cases = (
            [  # Linear propagator with padding.
                (  # Padding from 2 to 4
                    propagator := create_propagator.linear(
                        time_step_size=(dt := 1.0),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_p := 0.1), dimension=(D_init := 2)
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_u := 0.2), dimension=D_init
                            ),
                        ),
                    ),
                    padded_propagator := create_propagator.linear(
                        time_step_size=dt,
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_p, dimension=D_init
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_u, dimension=D_init
                            ),
                        ),
                        # A covariance array must not be singular in any dimension, hence the epsilon.
                        resizing=create_propagator.covariance.resize(
                            pad_to=(D_p := 4), epsilon=(epsilon := 0.001)
                        ),
                    ),
                    original_dimension := 2,  # position = x, y
                    expected_dimension := D_p,  # padded position = x, y, _, _
                    match_dimension := 2,  # first 2 dimensions should match
                    epsilon,
                ),
                (  # Padding to same dimension should be a no-op
                    propagator := create_propagator.linear(
                        time_step_size=(dt := 1.0),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_p := 0.1), dimension=(D_init := 2)
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_u := 0.2), dimension=D_init
                            ),
                        ),
                    ),
                    padded_propagator := create_propagator.linear(
                        time_step_size=dt,
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_p, dimension=D_init
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_u, dimension=D_init
                            ),
                        ),
                        # A covariance array must not be singular in any dimension, hence the epsilon.
                        resizing=create_propagator.covariance.resize(
                            pad_to=(D_p := 2), epsilon=(epsilon := 0.001)
                        ),
                    ),
                    original_dimension := 2,
                    expected_dimension := D_p,
                    match_dimension := 2,  # both are 2x2, should match completely
                    epsilon,
                ),
                (  # Extract X-coordinate + pad to 3
                    propagator := create_propagator.linear(
                        time_step_size=(dt := 1.0),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_p := 0.1), dimension=(D_init := 2)
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_u := 0.2), dimension=D_init
                            ),
                        ),
                    ),
                    extracted_and_padded_propagator := create_propagator.linear(
                        time_step_size=dt,
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_p, dimension=D_init
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_u, dimension=D_init
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            keep=ExtractXCoordinateCovariance(),
                            pad_to=(D_p := 3),
                            epsilon=(epsilon := 0.001),
                        ),
                    ),
                    original_dimension := 2,  # position = x, y
                    expected_dimension := D_p,  # extracted x, padded to 3
                    match_dimension := 1,  # only x should match
                    epsilon := epsilon,
                ),
            ]
            + [  # EKF propagator with covariance extraction and padding.
                (  # Pad to dimension 5
                    propagator := create_propagator.ekf(
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                        ),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_p := 0.1), dimension=(D_o := 4)
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_u := 0.2), dimension=(D_u := 2)
                            ),
                        ),
                    ),
                    unchanged_propagator := create_propagator.ekf(
                        model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_p, dimension=D_o
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_u, dimension=D_u
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            pad_to=(D_p := 5), epsilon=(epsilon := 0.001)
                        ),
                    ),
                    original_dimension := 4,  # state = x, y, heading, velocity
                    expected_dimension := 5,  # padded state = ...state, _
                    match_dimension := 4,  # original state should match
                    epsilon,
                ),
                (  # Extract pose from full state covariance
                    propagator := create_propagator.ekf(
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                        ),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_p := 0.1), dimension=(D_o := 4)
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_u := 0.2), dimension=(D_u := 2)
                            ),
                        ),
                    ),
                    padded_propagator := create_propagator.ekf(
                        model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_p, dimension=D_o
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_u, dimension=D_u
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            keep=model.bicycle.covariance_of.pose(),
                            epsilon=(epsilon := 0.001),
                        ),
                    ),
                    original_dimension := 4,  # state = x, y, heading, velocity
                    expected_dimension := 3,  # pose = x, y, heading
                    match_dimension := 3,  # first 3 dimensions should match (pose)
                    epsilon,
                ),
                (  # Padding to the same dimension should be a no-op
                    propagator := create_propagator.ekf(
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                        ),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_p := 0.1), dimension=(D_o := 4)
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_u := 0.2), dimension=(D_u := 2)
                            ),
                        ),
                    ),
                    padded_propagator := create_propagator.ekf(
                        model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_p, dimension=D_o
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_u, dimension=D_u
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            pad_to=(D_p := 4), epsilon=(epsilon := 0.001)
                        ),
                    ),
                    original_dimension := 4,  # state = x, y, heading, velocity
                    expected_dimension := D_p,  # state = x, y, heading, velocity
                    match_dimension := 4,  # full state should match
                    epsilon,
                ),
                (  # Extract position from state + pad to 3 (e.g. when ignoring heading uncertainty).
                    propagator := create_propagator.ekf(
                        model=model.bicycle.obstacle(
                            time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                        ),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_p := 0.1), dimension=(D_o := 4)
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=(sigma_u := 0.2), dimension=(D_u := 2)
                            ),
                        ),
                    ),
                    extracted_and_padded_propagator := create_propagator.ekf(
                        model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_p, dimension=D_o
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=sigma_u, dimension=D_u
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            keep=model.bicycle.covariance_of.position(),
                            pad_to=(D_p := 3),
                            epsilon=(epsilon := 0.001),
                        ),
                    ),
                    original_dimension := 4,  # state = x, y, heading, velocity
                    expected_dimension := D_p,  # extracted position (x, y), padded to 3
                    match_dimension := 2,  # only position should match
                    epsilon,
                ),
            ]
        )

        return [
            (
                predictor := create_predictor.curvilinear(
                    horizon=(T_p := 20),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                    ),
                    estimator=model.bicycle.estimator(time_step_size=dt, wheelbase=L),
                    prediction=prediction_creator.simple(
                        resize_states_to=original_dimension
                    ),
                    propagator=propagator,
                ),
                padded_predictor := create_predictor.curvilinear(
                    horizon=(T_p := 20),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                    ),
                    estimator=model.bicycle.estimator(time_step_size=dt, wheelbase=L),
                    prediction=prediction_creator.simple(
                        resize_states_to=expected_dimension
                    ),
                    propagator=padded_propagator,
                ),
                history := data.obstacle_2d_poses(
                    x=array(
                        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], shape=(T := 3, K := 2)
                    ),
                    y=array([[6.0, 1.0], [4.0, 3.0], [2.0, 5.0]], shape=(T, K)),
                ),
                original_dimension,
                expected_dimension,
                match_dimension,
                epsilon,
            )
            for propagator, padded_propagator, original_dimension, expected_dimension, match_dimension, epsilon in cases
        ]

    @mark.parametrize(
        [
            "predictor",
            "padded_predictor",
            "history",
            "original_dimension",
            "expected_dimension",
            "match_dimension",
            "epsilon",
        ],
        [
            *cases(
                create_propagator=create_propagator.numpy,
                create_predictor=create_predictor.numpy,
                data=data.numpy,
                model=model.numpy,
                prediction_creator=prediction_creator.numpy,
            ),
            *cases(
                create_propagator=create_propagator.jax,
                create_predictor=create_predictor.jax,
                data=data.jax,
                model=model.jax,
                prediction_creator=prediction_creator.jax,
            ),
        ],
    )
    def test[HistoryT](
        self,
        predictor: ObstacleMotionPredictor[HistoryT, ObstacleStates],
        padded_predictor: ObstacleMotionPredictor[HistoryT, ObstacleStates],
        history: HistoryT,
        original_dimension: int,
        expected_dimension: int,
        match_dimension: int,
        epsilon: float,
    ) -> None:
        covariances = predictor.predict(history=history).covariance()
        padded_covariances = padded_predictor.predict(history=history).covariance()

        D_m = match_dimension
        T, D_o, _, K = covariances.shape
        T_p, D_p, _, K_p = padded_covariances.shape

        assert (T, K) == (T_p, K_p), (
            f"Expected {T} time steps and {K} obstacles, got {T_p} and {K_p}."
        )
        assert D_o == original_dimension, (
            f"Expected original dimension {original_dimension}, got {D_o}."
        )
        assert D_p == expected_dimension, (
            f"Expected padded dimension {expected_dimension}, got {D_p}."
        )

        assert np.allclose(
            padded_covariances[:, :D_m, :D_m, :], covariances[:, :D_m, :D_m, :]
        ), "Covariance submatrices do not match within the expected dimension range."

        check.has_diagonal_padding(
            padded_covariances, from_dimension=D_m, epsilon=epsilon
        )


class test_that_covariance_is_always_symmetric_and_positive_semidefinite:
    @staticmethod
    def cases(
        create_predictor, model, create_propagator, prediction_creator, data
    ) -> Sequence[tuple]:
        dt = 0.1
        L = 1.0
        return [
            (
                create_predictor.curvilinear(
                    horizon=20,
                    model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                    estimator=model.bicycle.estimator(time_step_size=dt, wheelbase=L),
                    prediction=prediction_creator.bicycle(),
                    propagator=propagator,
                ),
                data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.5]], shape=(2, 1)),
                    heading=array([[0.0], [0.4]], shape=(2, 1)),
                ),
            )
            for propagator in [
                create_propagator.ekf(
                    model=model.bicycle.obstacle(time_step_size=dt, wheelbase=1.0),
                    covariance=create_propagator.covariance.composite(
                        state_provider=create_propagator.covariance.constant_variance(
                            variance=0.1, dimension=4
                        ),
                        input_provider=create_propagator.covariance.constant_variance(
                            variance=0.2, dimension=2
                        ),
                    ),
                    resizing=create_propagator.covariance.resize(
                        keep=model.bicycle.covariance_of.pose(), epsilon=1e-15
                    ),
                ),
                create_propagator.linear(
                    time_step_size=dt,
                    covariance=create_propagator.covariance.composite(
                        state_provider=create_propagator.covariance.constant_variance(
                            variance=0.1, dimension=2
                        ),
                        input_provider=create_propagator.covariance.constant_variance(
                            variance=0.2, dimension=2
                        ),
                    ),
                    resizing=create_propagator.covariance.resize(
                        pad_to=3, epsilon=1e-15
                    ),
                ),
            ]
        ]

    @mark.parametrize(
        ["predictor", "history"],
        [
            *cases(
                create_predictor=create_predictor.numpy,
                model=model.numpy,
                create_propagator=create_propagator.numpy,
                prediction_creator=prediction_creator.numpy,
                data=data.numpy,
            ),
            *cases(
                create_predictor=create_predictor.jax,
                model=model.jax,
                create_propagator=create_propagator.jax,
                prediction_creator=prediction_creator.jax,
                data=data.jax,
            ),
        ],
    )
    def test[HistoryT](
        self,
        predictor: ObstacleMotionPredictor[HistoryT, ObstacleStates],
        history: HistoryT,
    ) -> None:
        assert check.is_spd(predictor.predict(history=history).covariance())


class test_that_different_backends_produce_matching_results:
    @staticmethod
    def cases() -> Sequence:
        dt = 0.1
        L = 1.0
        T_p = 10

        def linear[HistoryT](
            create_predictor, model, create_propagator, prediction_creator, data
        ) -> tuple[ObstacleMotionPredictor[HistoryT, ObstacleStates], HistoryT]:
            return (
                create_predictor.curvilinear(
                    horizon=T_p,
                    model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                    estimator=model.bicycle.estimator(time_step_size=dt, wheelbase=L),
                    prediction=prediction_creator.bicycle(),
                    propagator=create_propagator.linear(
                        time_step_size=dt,
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=0.1, dimension=2
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=0.2, dimension=2
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            pad_to=3, epsilon=1e-15
                        ),
                    ),
                ),
                data.obstacle_2d_poses(
                    x=array([[0.0], [1.0], [2.0]], shape=(3, 1)),
                    y=array([[0.0], [0.5], [1.0]], shape=(3, 1)),
                ),
            )

        def ekf[HistoryT](
            create_predictor, model, create_propagator, prediction_creator, data
        ) -> tuple[ObstacleMotionPredictor[HistoryT, ObstacleStates], HistoryT]:
            return (
                create_predictor.curvilinear(
                    horizon=T_p,
                    model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                    estimator=model.bicycle.estimator(time_step_size=dt, wheelbase=L),
                    prediction=prediction_creator.bicycle(),
                    propagator=create_propagator.ekf(
                        model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=0.1, dimension=4
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=0.2, dimension=2
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            keep=model.bicycle.covariance_of.pose(), epsilon=1e-15
                        ),
                    ),
                ),
                data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.5]], shape=(2, 1)),
                    heading=array([[0.0], [0.2]], shape=(2, 1)),
                ),
            )

        return [
            [
                predictor(
                    create_predictor=create_predictor.numpy,
                    model=model.numpy,
                    create_propagator=create_propagator.numpy,
                    prediction_creator=prediction_creator.numpy,
                    data=data.numpy,
                ),
                predictor(
                    create_predictor=create_predictor.jax,
                    model=model.jax,
                    create_propagator=create_propagator.jax,
                    prediction_creator=prediction_creator.jax,
                    data=data.jax,
                ),
            ]
            for predictor in [linear, ekf]
        ]

    @mark.parametrize("predictors", cases())
    def test[HistoryT](
        self,
        predictors: Sequence[
            tuple[ObstacleMotionPredictor[HistoryT, ObstacleStates], HistoryT]
        ],
    ) -> None:
        covariances = [
            predictor.predict(history=history).covariance()
            for predictor, history in predictors
        ]

        reference = covariances[0]

        for i, covariance in enumerate(covariances[1:], start=1):
            assert np.allclose(reference, covariance, atol=1e-4), (
                f"Covariance from predictor {i} does not match reference covariance from predictor 0. "
                f"Reference covariance: {reference}, covariance {i}: {covariance}"
            )
