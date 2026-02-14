from typing import Sequence, Protocol

from trajax import ObstacleStateEstimator, EstimatedObstacleStates, model

from numtypes import Array, Dims, array

import numpy as np

from tests.dsl import ArrayConvertible, mppi as data, check
from pytest import mark, Subtests


class EstimatorCreator[HistoryT, StatesT, InputsT](Protocol):
    def __call__(
        self, *, process_noise: float, observation_noise: float
    ) -> ObstacleStateEstimator[HistoryT, StatesT, InputsT, ArrayConvertible]:
        """Creates an obstacle state estimator with the given noise covariances."""
        ...


class test_that_covariance_satisfies_basic_properties:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1

        def random_2d_poses(T: int, K: int):
            return data.obstacle_2d_poses(
                x=array(np.random.randn(T, K), shape=(T, K)),
                y=array(np.random.randn(T, K), shape=(T, K)),
                heading=array(np.random.randn(T, K), shape=(T, K)),
            )

        def random_simple_obstacle_states(T: int, D_o: int, K: int):
            return data.simple_obstacle_states(
                states=array(np.random.randn(T, D_o, K), shape=(T, D_o, K)),
            )

        return [
            (
                estimator := model.integrator.estimator.kf(
                    time_step_size=dt,
                    process_noise_covariance=1e-5,
                    observation_noise_covariance=1e-5,
                    observation_dimension=(D_o := 4),
                ),
                history := random_simple_obstacle_states(T := 5, D_o, K := 2),
                expected_shape := (2 * D_o, 2 * D_o, K),
            ),
            *[
                (
                    estimator,
                    history := random_2d_poses(T := 6, K := 5),
                    expected_shape := (4 + 2, 4 + 2, K),
                )
                for estimator in [
                    model.bicycle.estimator.ekf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.bicycle.estimator.ukf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                ]
            ],
            *[
                (
                    estimator,
                    history := random_2d_poses(T := 2, K := 4),
                    expected_shape := (3 + 2, 3 + 2, K),
                )
                for estimator in [
                    model.unicycle.estimator.ekf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.ukf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                ]
            ],
        ]

    @mark.parametrize(
        ["estimator", "history", "expected_shape"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        subtests: Subtests,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT, ArrayConvertible],
        history: HistoryT,
        expected_shape: tuple[int, int, int],
    ) -> None:
        result = estimator.estimate_from(history)
        covariance = np.asarray(result.covariance)

        with subtests.test("covariance is not None"):
            assert result.covariance is not None

        with subtests.test("covariance has correct shape"):
            assert covariance.shape == expected_shape

        with subtests.test("covariance is symmetric positive definite"):
            assert check.is_spd(covariance)


class test_that_covariance_decreases_with_more_observations:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        rng = np.random.default_rng(32)
        velocities = rng.random(size=(D_o := 3, K := 4)) * 0.5
        velocities_x, velocities_y = rng.random(size=(2, K)) * 0.5
        angular_velocities = rng.random(size=(K,)) * 0.1

        return [
            (
                estimator := model.integrator.estimator.kf(
                    time_step_size=dt,
                    process_noise_covariance=1e-3,
                    observation_noise_covariance=1e-3,
                    observation_dimension=D_o,
                ),
                short_history := data.simple_obstacle_states(
                    states=array(
                        shape=(T := 3, D_o, K),
                        elements=[dt * t * velocities for t in range(T)],
                    ),
                ),
                long_history := data.simple_obstacle_states(
                    states=array(
                        shape=(T := 10, D_o, K),
                        elements=[dt * t * velocities for t in range(T)],
                    ),
                ),
            ),
            *[
                (
                    estimator,
                    short_history := data.obstacle_2d_poses(
                        x=array(
                            shape=(T := 3, K),
                            elements=[dt * t * velocities_x for t in range(T)],
                        ),
                        y=array(
                            shape=(T, K),
                            elements=[dt * t * velocities_y for t in range(T)],
                        ),
                        heading=array(
                            shape=(T, K),
                            elements=[dt * t * angular_velocities for t in range(T)],
                        ),
                    ),
                    long_history := data.obstacle_2d_poses(
                        x=array(
                            shape=(T := 10, K),
                            elements=[dt * t * velocities_x for t in range(T)],
                        ),
                        y=array(
                            shape=(T, K),
                            elements=[dt * t * velocities_y for t in range(T)],
                        ),
                        heading=array(
                            shape=(T, K),
                            elements=[dt * t * angular_velocities for t in range(T)],
                        ),
                    ),
                )
                for estimator in [
                    model.bicycle.estimator.ekf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-3,
                        observation_noise_covariance=1e-3,
                    ),
                    model.bicycle.estimator.ukf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-3,
                        observation_noise_covariance=1e-3,
                    ),
                    model.unicycle.estimator.ekf(
                        time_step_size=dt,
                        process_noise_covariance=1e-3,
                        observation_noise_covariance=1e-3,
                    ),
                    model.unicycle.estimator.ukf(
                        time_step_size=dt,
                        process_noise_covariance=1e-3,
                        observation_noise_covariance=1e-3,
                    ),
                ]
            ],
        ]

    @mark.parametrize(
        ["estimator", "short_history", "long_history"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        short_history: HistoryT,
        long_history: HistoryT,
    ) -> None:
        short_result = estimator.estimate_from(short_history)
        long_result = estimator.estimate_from(long_history)

        assert np.linalg.norm(short_result.covariance) > np.linalg.norm(
            long_result.covariance
        ), "Covariance should decrease with more observations."


class test_that_covariance_is_larger_for_unobserved_states:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        T = 5
        K = 2
        rng = np.random.default_rng(42)

        return [
            (
                estimator := model.integrator.estimator.kf(
                    time_step_size=dt,
                    process_noise_covariance=1e-5,
                    observation_noise_covariance=1e-5,
                    observation_dimension=(D_o := 2),
                ),
                history := data.simple_obstacle_states(
                    states=array(
                        rng.standard_normal(size=(T, D_o, K)), shape=(T, D_o, K)
                    ),
                ),
                position_indices := slice(0, D_o),
                velocity_indices := slice(D_o, 2 * D_o),
            ),
            *[
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array(rng.standard_normal(size=(T, K)), shape=(T, K)),
                        y=array(rng.standard_normal(size=(T, K)), shape=(T, K)),
                        heading=array(rng.standard_normal(size=(T, K)), shape=(T, K)),
                    ),
                    position_indices := slice(0, 2),
                    velocity_indices := slice(3, 5),
                )
                for estimator in [
                    model.bicycle.estimator.ekf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.bicycle.estimator.ukf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.ekf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.ukf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                ]
            ],
        ]

    @mark.parametrize(
        ["estimator", "history", "observed_indices", "unobserved_indices"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        history: HistoryT,
        observed_indices: slice,
        unobserved_indices: slice,
    ) -> None:
        result = estimator.estimate_from(history)
        covariance = np.asarray(result.covariance)

        observed_cov = covariance[observed_indices, observed_indices, 0]
        unobserved_cov = covariance[unobserved_indices, unobserved_indices, 0]

        assert np.trace(unobserved_cov) > np.trace(observed_cov)


class test_that_higher_noise_yields_higher_covariance:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        T = 5
        K = 2
        rng = np.random.default_rng(42)

        return [
            (
                estimator := lambda process_noise, observation_noise: (
                    model.integrator.estimator.kf(
                        time_step_size=dt,
                        process_noise_covariance=process_noise,
                        observation_noise_covariance=observation_noise,
                        observation_dimension=(D_o := 2),
                    )
                ),
                history := data.simple_obstacle_states(
                    states=array(
                        rng.standard_normal(size=(T, D_o := 2, K)), shape=(T, D_o, K)
                    ),
                ),
            ),
            *[
                (
                    estimator,
                    history := data.obstacle_2d_poses(
                        x=array(rng.standard_normal(size=(T, K)), shape=(T, K)),
                        y=array(rng.standard_normal(size=(T, K)), shape=(T, K)),
                        heading=array(rng.standard_normal(size=(T, K)), shape=(T, K)),
                    ),
                )
                for estimator in [
                    lambda process_noise, observation_noise, dt=dt: (
                        model.bicycle.estimator.ekf(
                            time_step_size=dt,
                            wheelbase=1.0,
                            process_noise_covariance=process_noise,
                            observation_noise_covariance=observation_noise,
                        )
                    ),
                    lambda process_noise, observation_noise, dt=dt: (
                        model.bicycle.estimator.ukf(
                            time_step_size=dt,
                            wheelbase=1.0,
                            process_noise_covariance=process_noise,
                            observation_noise_covariance=observation_noise,
                        )
                    ),
                    lambda process_noise, observation_noise, dt=dt: (
                        model.unicycle.estimator.ekf(
                            time_step_size=dt,
                            process_noise_covariance=process_noise,
                            observation_noise_covariance=observation_noise,
                        )
                    ),
                    lambda process_noise, observation_noise, dt=dt: (
                        model.unicycle.estimator.ukf(
                            time_step_size=dt,
                            process_noise_covariance=process_noise,
                            observation_noise_covariance=observation_noise,
                        )
                    ),
                ]
            ],
        ]

    @mark.parametrize(
        ["estimator", "history"],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        subtests: Subtests,
        estimator: EstimatorCreator[HistoryT, StatesT, InputsT],
        history: HistoryT,
    ) -> None:
        def estimate_with(
            *, process_noise: float, observation_noise: float
        ) -> EstimatedObstacleStates[StatesT, InputsT, ArrayConvertible]:
            return estimator(
                process_noise=process_noise, observation_noise=observation_noise
            ).estimate_from(history)

        with subtests.test("higher process noise yields higher covariance"):
            low_result = estimate_with(process_noise=1e-5, observation_noise=1e-5)
            high_result = estimate_with(process_noise=1.0, observation_noise=1e-5)

            for low, high in zip(
                np.asarray(low_result.covariance).transpose(2, 0, 1),
                np.asarray(high_result.covariance).transpose(2, 0, 1),
            ):
                assert np.trace(high) > np.trace(low)

        with subtests.test("higher observation noise yields higher covariance"):
            low_result = estimate_with(process_noise=1e-5, observation_noise=1e-5)
            high_result = estimate_with(process_noise=1e-5, observation_noise=1.0)

            for low, high in zip(
                np.asarray(low_result.covariance).transpose(2, 0, 1),
                np.asarray(high_result.covariance).transpose(2, 0, 1),
            ):
                assert np.trace(high) > np.trace(low)


class test_that_covariance_is_independent_across_obstacles:
    @staticmethod
    def cases(model, data) -> Sequence[tuple]:
        dt = 0.1
        T = 5
        K = 3
        rng = np.random.default_rng(42)

        def perturb[T: int, K: int](
            poses: Array[Dims[T, K]], *, index: int
        ) -> Array[Dims[T, K]]:
            perturbed = poses.copy()
            perturbed[:, index] += 10.0
            return perturbed

        def perturb_simple[T: int, D_o: int, K: int](
            states: Array[Dims[T, D_o, K]], *, index: int
        ) -> Array[Dims[T, D_o, K]]:
            perturbed = states.copy()
            perturbed[:, :, index] += 10.0
            return perturbed

        return [
            (
                estimator := model.integrator.estimator.kf(
                    time_step_size=dt,
                    process_noise_covariance=1e-5,
                    observation_noise_covariance=1e-5,
                    observation_dimension=(D_o := 2),
                ),
                base_history := data.simple_obstacle_states(
                    states=array(
                        base := rng.standard_normal(size=(T, D_o, K)), shape=(T, D_o, K)
                    ),
                ),
                perturbed_history := data.simple_obstacle_states(
                    states=array(perturb_simple(base, index=0), shape=(T, D_o, K)),
                ),
                unperturbed_index := 1,
            ),
            *[
                (
                    estimator,
                    base_history := data.obstacle_2d_poses(
                        x=array(
                            base_x := rng.standard_normal(size=(T, K)), shape=(T, K)
                        ),
                        y=array(
                            base_y := rng.standard_normal(size=(T, K)), shape=(T, K)
                        ),
                        heading=array(
                            base_heading := rng.standard_normal(size=(T, K)),
                            shape=(T, K),
                        ),
                    ),
                    perturbed_history := data.obstacle_2d_poses(
                        x=array(base_x, shape=(T, K)),
                        y=array(base_y, shape=(T, K)),
                        heading=array(perturb(base_heading, index=1), shape=(T, K)),
                    ),
                    unperturbed_index := 0,
                )
                for estimator in [
                    model.bicycle.estimator.ekf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.bicycle.estimator.ukf(
                        time_step_size=dt,
                        wheelbase=1.0,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.ekf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                    model.unicycle.estimator.ukf(
                        time_step_size=dt,
                        process_noise_covariance=1e-5,
                        observation_noise_covariance=1e-5,
                    ),
                ]
            ],
        ]

    @mark.parametrize(
        [
            "estimator",
            "base_history",
            "perturbed_history",
            "unperturbed_index",
        ],
        [
            *cases(model=model.numpy, data=data.numpy),
            *cases(model=model.jax, data=data.jax),
        ],
    )
    def test[HistoryT, StatesT, InputsT](
        self,
        estimator: ObstacleStateEstimator[HistoryT, StatesT, InputsT],
        base_history: HistoryT,
        perturbed_history: HistoryT,
        unperturbed_index: int,
    ) -> None:
        k = unperturbed_index
        base_result = estimator.estimate_from(base_history)
        perturbed_result = estimator.estimate_from(perturbed_history)

        base_cov = np.asarray(base_result.covariance)
        perturbed_cov = np.asarray(perturbed_result.covariance)

        assert np.allclose(base_cov[:, :, k], perturbed_cov[:, :, k], atol=1e-10)
