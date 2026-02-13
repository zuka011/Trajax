from typing import Sequence, Any

from trajax import (
    ObstacleModel,
    ObstacleMotionPredictor,
    ObstacleStates,
    model,
    types,
    propagator as create_propagator,
    predictor as create_predictor,
)

from numtypes import array
from jaxtyping import ArrayLike

import numpy as np

from tests.dsl import mppi as data, prediction_creator
from tests.dsl.numeric import (
    ObstacleStatesWrapper,
    ObstacleControlInputSequencesWrapper,
    compute,
    estimate,
)
from pytest import mark


class test_that_jacobians_match_finite_differences:
    @staticmethod
    def cases(model, types, epsilon) -> Sequence[tuple]:

        def bicycle_to_states(states):
            return types.bicycle.obstacle_states.wrap(states)

        def bicycle_to_inputs(inputs):
            return types.bicycle.obstacle_control_input_sequences.wrap(inputs)

        def unicycle_to_states(states):
            return types.unicycle.obstacle_states.wrap(states)

        def unicycle_to_inputs(inputs):
            return types.unicycle.obstacle_control_input_sequences.wrap(inputs)

        return [
            (  # Bicycle: single obstacle
                model.bicycle.obstacle(time_step_size=0.1, wheelbase=2.5),
                types.bicycle.obstacle_state_sequences.create(
                    x=array([[1.0]], shape=(T := 1, K := 1)),
                    y=array([[2.0]], shape=(T, K)),
                    heading=array([[0.3]], shape=(T, K)),
                    speed=array([[5.0]], shape=(T, K)),
                ),
                types.bicycle.obstacle_control_input_sequences.create(
                    accelerations=array([[0.0]], shape=(T, K)),
                    steering_angles=array([[0.1]], shape=(T, K)),
                ),
                bicycle_to_states,
                bicycle_to_inputs,
                epsilon,
            ),
            (  # Bicycle: multiple obstacles
                model.bicycle.obstacle(time_step_size=0.1, wheelbase=1.0),
                types.bicycle.obstacle_state_sequences.create(
                    x=array([[0.0, 5.0]], shape=(T := 1, K := 2)),
                    y=array([[0.0, 3.0]], shape=(T, K)),
                    heading=array([[0.0, np.pi / 4]], shape=(T, K)),
                    speed=array([[10.0, 3.0]], shape=(T, K)),
                ),
                types.bicycle.obstacle_control_input_sequences.create(
                    accelerations=array([[0.0, 0.0]], shape=(T, K)),
                    steering_angles=array([[0.2, -0.1]], shape=(T, K)),
                ),
                bicycle_to_states,
                bicycle_to_inputs,
                epsilon,
            ),
            (  # Unicycle: single obstacle
                model.unicycle.obstacle(time_step_size=0.1),
                types.unicycle.obstacle_state_sequences.create(
                    x=array([[1.0]], shape=(T := 1, K := 1)),
                    y=array([[2.0]], shape=(T, K)),
                    heading=array([[0.3]], shape=(T, K)),
                ),
                types.unicycle.obstacle_control_input_sequences.create(
                    linear_velocities=array([[5.0]], shape=(T, K)),
                    angular_velocities=array([[0.1]], shape=(T, K)),
                ),
                unicycle_to_states,
                unicycle_to_inputs,
                epsilon,
            ),
            (  # Unicycle: multiple obstacles
                model.unicycle.obstacle(time_step_size=0.1),
                types.unicycle.obstacle_state_sequences.create(
                    x=array([[0.0, 5.0]], shape=(T := 1, K := 2)),
                    y=array([[0.0, 3.0]], shape=(T, K)),
                    heading=array([[0.0, np.pi / 4]], shape=(T, K)),
                ),
                types.unicycle.obstacle_control_input_sequences.create(
                    linear_velocities=array([[10.0, 3.0]], shape=(T, K)),
                    angular_velocities=array([[0.2, -0.1]], shape=(T, K)),
                ),
                unicycle_to_states,
                unicycle_to_inputs,
                epsilon,
            ),
        ]

    @mark.parametrize(
        ["obstacle_model", "states", "inputs", "to_states", "to_inputs", "epsilon"],
        [
            *cases(model=model.numpy, types=types.numpy, epsilon=1e-6),
            *cases(model=model.jax, types=types.jax, epsilon=1e-4),
        ],
    )
    def test[InputSequencesT, StateSequencesT, JacobianT: ArrayLike](
        self,
        obstacle_model: ObstacleModel[
            Any, Any, Any, InputSequencesT, StateSequencesT, JacobianT
        ],
        states: StateSequencesT,
        inputs: InputSequencesT,
        to_states: ObstacleStatesWrapper[StateSequencesT],
        to_inputs: ObstacleControlInputSequencesWrapper[InputSequencesT],
        epsilon: float,
    ) -> None:
        state_jacobian = obstacle_model.state_jacobian(states=states, inputs=inputs)
        state_jacobian_finite_diff = estimate.state_jacobian(
            obstacle_model,
            states=states,
            inputs=inputs,
            to_states=to_states,
            to_inputs=to_inputs,
            epsilon=epsilon,
        )

        assert np.allclose(
            state_jacobian, state_jacobian_finite_diff, rtol=1e-2, atol=1e-2
        ), (
            f"State Jacobian mismatch. Analytical:\n{state_jacobian}\n"
            f"Finite difference:\n{state_jacobian_finite_diff}"
        )

        input_jacobian = obstacle_model.input_jacobian(states=states, inputs=inputs)
        input_jacobian_finite_diff = estimate.input_jacobian(
            obstacle_model,
            states=states,
            inputs=inputs,
            to_states=to_states,
            to_inputs=to_inputs,
            epsilon=epsilon,
        )

        assert np.allclose(
            input_jacobian, input_jacobian_finite_diff, rtol=1e-2, atol=1e-2
        ), (
            f"Input Jacobian mismatch. Analytical:\n{input_jacobian}\n"
            f"Finite difference:\n{input_jacobian_finite_diff}"
        )


class test_that_ekf_covariance_is_more_isotropic_when_turning_sharply:
    @staticmethod
    def cases(
        create_predictor,
        model,
        create_propagator,
        bicycle_prediction_creator,
        data,
    ) -> Sequence[tuple]:
        return [
            (
                predictor := create_predictor.curvilinear(
                    horizon=(T_p := 10),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                    ),
                    estimator=model.bicycle.estimator(time_step_size=dt, wheelbase=L),
                    prediction=bicycle_prediction_creator(),
                    propagator=create_propagator.ekf(
                        model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=0.1, dimension=4
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=0.1, dimension=2
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            keep=model.bicycle.covariance_of.pose(), epsilon=1e-15
                        ),
                    ),
                ),
                # Straight history: v=10, δ=0
                straight := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.0]], shape=(2, 1)),
                    heading=array([[0.0], [0.0]], shape=(2, 1)),
                ),
                # Turning history: v=10, heading changes
                turning := data.obstacle_2d_poses(
                    x=array([[0.0], [1.0]], shape=(2, 1)),
                    y=array([[0.0], [0.0]], shape=(2, 1)),
                    heading=array([[0.0], [0.4]], shape=(2, 1)),
                ),
            ),
        ]

    @mark.parametrize(
        ["predictor", "straight_history", "turning_history"],
        [
            *cases(
                create_predictor=create_predictor.numpy,
                model=model.numpy,
                create_propagator=create_propagator.numpy,
                bicycle_prediction_creator=prediction_creator.numpy.bicycle,
                data=data.numpy,
            ),
            *cases(
                create_predictor=create_predictor.jax,
                model=model.jax,
                create_propagator=create_propagator.jax,
                bicycle_prediction_creator=prediction_creator.jax.bicycle,
                data=data.jax,
            ),
        ],
    )
    def test[HistoryT](
        self,
        predictor: ObstacleMotionPredictor[HistoryT, ObstacleStates],
        straight_history: HistoryT,
        turning_history: HistoryT,
    ) -> None:
        straight = predictor.predict(history=straight_history).covariance()
        turning = predictor.predict(history=turning_history).covariance()

        assert (
            turning_condition := compute.condition_number(turning[-1, :2, :2, 0])
        ) < (straight_condition := compute.condition_number(straight[-1, :2, :2, 0])), (
            f"Expected turning covariance to be more isotropic than straight, but got condition numbers "
            f"{turning_condition:.2f} (turning) vs {straight_condition:.2f} (straight)."
        )


class test_that_ekf_matches_linear_for_stationary_obstacles:
    @staticmethod
    def cases(
        create_predictor, model, create_propagator, unicycle_prediction_creator, data
    ) -> Sequence[tuple]:
        return [
            (
                ekf := create_predictor.curvilinear(
                    horizon=(T_p := 5),
                    model=model.unicycle.obstacle(time_step_size=(dt := 0.1)),
                    estimator=model.unicycle.estimator(time_step_size=dt),
                    prediction=unicycle_prediction_creator(),
                    propagator=create_propagator.ekf(
                        model=model.unicycle.obstacle(time_step_size=dt),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=0.1, dimension=3
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=0.0, dimension=2
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            pad_to=3, epsilon=1e-15
                        ),
                    ),
                ),
                linear := create_predictor.curvilinear(
                    horizon=T_p,
                    model=model.unicycle.obstacle(time_step_size=dt),
                    estimator=model.unicycle.estimator(time_step_size=dt),
                    prediction=unicycle_prediction_creator(),
                    propagator=create_propagator.linear(
                        time_step_size=dt,
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=0.1, dimension=3
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=0.0, dimension=3
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            pad_to=3, epsilon=1e-15
                        ),
                    ),
                ),
                # Stationary history: θ=0, v=0, ω=0
                history := data.obstacle_2d_poses(
                    x=array([[0.0], [0.0]], shape=(T := 2, K := 1)),
                    y=array([[0.0], [0.0]], shape=(T, K)),
                    heading=array([[0.0], [0.0]], shape=(T, K)),
                ),
            ),
        ]

    @mark.parametrize(
        ["ekf", "linear", "history"],
        [
            *cases(
                create_predictor=create_predictor.numpy,
                model=model.numpy,
                create_propagator=create_propagator.numpy,
                unicycle_prediction_creator=prediction_creator.numpy.unicycle,
                data=data.numpy,
            ),
            *cases(
                create_predictor=create_predictor.jax,
                model=model.jax,
                create_propagator=create_propagator.jax,
                unicycle_prediction_creator=prediction_creator.jax.unicycle,
                data=data.jax,
            ),
        ],
    )
    def test[HistoryT](
        self,
        ekf: ObstacleMotionPredictor[HistoryT, ObstacleStates],
        linear: ObstacleMotionPredictor[HistoryT, ObstacleStates],
        history: HistoryT,
    ) -> None:
        assert np.allclose(
            ekf.predict(history=history).covariance(),
            linear.predict(history=history).covariance(),
            atol=1e-6,
        )


class test_that_ekf_covariance_depends_on_obstacle_heading:
    @staticmethod
    def cases(
        create_predictor,
        model,
        create_propagator,
        bicycle_prediction_creator,
        data,
    ) -> Sequence[tuple]:
        return [
            (
                create_predictor.curvilinear(
                    horizon=(T_p := 10),
                    model=model.bicycle.obstacle(
                        time_step_size=(dt := 0.1), wheelbase=(L := 1.0)
                    ),
                    estimator=model.bicycle.estimator(time_step_size=dt, wheelbase=L),
                    prediction=bicycle_prediction_creator(),
                    propagator=create_propagator.ekf(
                        model=model.bicycle.obstacle(time_step_size=dt, wheelbase=L),
                        covariance=create_propagator.covariance.composite(
                            state_provider=create_propagator.covariance.constant_variance(
                                variance=0.1, dimension=4
                            ),
                            input_provider=create_propagator.covariance.constant_variance(
                                variance=0.1, dimension=2
                            ),
                        ),
                        resizing=create_propagator.covariance.resize(
                            keep=model.bicycle.covariance_of.pose(), epsilon=1e-15
                        ),
                    ),
                ),
                data.obstacle_2d_poses(
                    x=array([[0.0, 0.0], [1.0, 1.0]], shape=(T := 2, K := 2)),
                    y=array([[0.0, 0.0], [0.0, 0.0]], shape=(T, K)),
                    # Obstacle 0 goes straight, obstacle 1 turns.
                    heading=array([[0.0, np.pi / 3], [0.0, np.pi / 3]], shape=(T, K)),
                ),
            ),
        ]

    @mark.parametrize(
        ["predictor", "history"],
        [
            *cases(
                create_predictor=create_predictor.numpy,
                model=model.numpy,
                create_propagator=create_propagator.numpy,
                bicycle_prediction_creator=prediction_creator.numpy.bicycle,
                data=data.numpy,
            ),
            *cases(
                create_predictor=create_predictor.jax,
                model=model.jax,
                create_propagator=create_propagator.jax,
                bicycle_prediction_creator=prediction_creator.jax.bicycle,
                data=data.jax,
            ),
        ],
    )
    def test[HistoryT](
        self,
        predictor: ObstacleMotionPredictor[HistoryT, ObstacleStates],
        history: HistoryT,
    ) -> None:
        covariances = np.asarray(predictor.predict(history=history).covariance())

        # The two obstacles have different headings, so their covariances diverge
        obstacle_0_cov = covariances[-1, :2, :2, 0]
        obstacle_1_cov = covariances[-1, :2, :2, 1]

        assert not np.allclose(obstacle_0_cov, obstacle_1_cov, atol=1e-3)
