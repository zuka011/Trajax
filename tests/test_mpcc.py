from typing import Protocol

from trajax import (
    AugmentedState,
    StateSequence,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
    DynamicalModel,
    Trajectory,
    ObstacleStates,
    ObstacleStateObserver,
    ObstacleSimulator,
    Weights,
    Mppi,
    MetricRegistry,
    CollisionMetric,
    MpccErrorMetric,
    access,
)
from trajax_visualizer import visualizer, MpccSimulationResult


from tests.visualize import VisualizationData
from tests.examples import mpcc, reference, obstacles
from pytest import mark


class MpccPlannerConfiguration[
    StateT: AugmentedState,
    StateBatchT: StateBatch,
    InputSequenceT: ControlInputSequence,
    InputBatchT: ControlInputBatch,
    ObstacleStatesForTimeStepT: ObstacleStates,
](Protocol):
    @property
    def reference(self) -> Trajectory:
        """Returns the reference trajectory."""
        ...

    @property
    def planner(self) -> Mppi[StateT, InputSequenceT, Weights]:
        """Returns the MPCC planner."""
        ...

    @property
    def model(
        self,
    ) -> DynamicalModel[
        StateT, StateSequence, StateBatchT, InputSequenceT, InputBatchT
    ]:
        """Returns the augmented dynamical model."""
        ...

    @property
    def wheelbase(self) -> float:
        """Returns the vehicle wheelbase (to be used in the visualization)."""
        ...

    @property
    def registry(self) -> MetricRegistry:
        """Returns the metric registry used by the planner."""
        ...

    @property
    def metrics(self) -> tuple[MpccErrorMetric, CollisionMetric | None]:
        """Returns the metrics used by the planner."""
        ...

    @property
    def obstacle_simulator(
        self,
    ) -> ObstacleSimulator[ObstacleStatesForTimeStepT] | None:
        """Returns the obstacle simulator, if any."""
        ...

    @property
    def obstacle_state_observer(
        self,
    ) -> ObstacleStateObserver[ObstacleStatesForTimeStepT] | None:
        """Returns the obstacle state observer, if any."""
        ...

    @property
    def initial_state(self) -> StateT:
        """Returns the initial augmented state."""
        ...

    @property
    def nominal_input(self) -> InputSequenceT:
        """Returns the nominal control input sequence."""
        ...


@mark.parametrize(
    ["configuration", "configuration_name"],
    [
        (mpcc.numpy.planner_from_base(), "numpy-from-base"),
        (mpcc.numpy.planner_from_augmented(), "numpy-from-augmented"),
        (mpcc.numpy.planner_from_mpcc(), "numpy-from-mpcc"),
        (
            mpcc.numpy.planner_from_mpcc(
                reference=reference.numpy.short_loop, use_boundary=True
            ),
            "numpy-from-mpcc-with-boundary",
        ),
        (mpcc.jax.planner_from_base(), "jax-from-base"),
        (mpcc.jax.planner_from_augmented(), "jax-from-augmented"),
        (mpcc.jax.planner_from_mpcc(), "jax-from-mpcc"),
        (
            mpcc.jax.planner_from_mpcc(
                reference=reference.jax.short_loop, use_boundary=True
            ),
            "jax-from-mpcc-with-boundary",
        ),
    ],
)
@mark.visualize.with_args(visualizer.mpcc(), lambda seed: seed)
@mark.filterwarnings("ignore:.*'obstacle_states'.*not.*data.*")
@mark.filterwarnings("error")
@mark.integration
def test_that_mpcc_planner_follows_trajectory_without_excessive_deviation[
    StateT: AugmentedState,
    StateBatchT: StateBatch,
    InputSequenceT: ControlInputSequence,
    InputBatchT: ControlInputBatch,
    ObstacleStatesT: ObstacleStates,
](
    visualization: VisualizationData[MpccSimulationResult],
    configuration: MpccPlannerConfiguration[
        StateT, StateBatchT, InputSequenceT, InputBatchT, ObstacleStatesT
    ],
    configuration_name: str,
) -> None:
    reference = configuration.reference
    planner = configuration.planner
    model = configuration.model
    wheelbase = configuration.wheelbase
    registry = configuration.registry
    error_metric, _ = configuration.metrics

    current_state = configuration.initial_state
    nominal_input = configuration.nominal_input

    min_progress = reference.path_length * 0.9
    progress = 0.0

    for t in range(max_steps := 150):
        control = planner.step(
            temperature=0.05, nominal_input=nominal_input, initial_state=current_state
        )

        nominal_input = control.nominal
        current_state = model.step(inputs=control.optimal, state=current_state)

        if (progress := current_state.virtual.array[0]) >= min_progress:
            break

    trajectories = registry.data(access.trajectories.require())
    errors = registry.get(error_metric)

    visualization.data_is(
        MpccSimulationResult(
            reference=reference,
            states=registry.data(access.states.require()),
            optimal_trajectories=[it.optimal for it in trajectories],
            nominal_trajectories=[it.nominal for it in trajectories],
            contouring_errors=errors.contouring,
            lag_errors=errors.lag,
            time_step_size=model.time_step_size,
            wheelbase=wheelbase,
            max_contouring_error=(max_contouring_error := 2.0),
            max_lag_error=(max_lag_error := 5.0),
        )
    ).seed_is(configuration_name)

    assert progress > min_progress, (
        f"Vehicle did not make sufficient progress along the path in {max_steps} steps. "
        f"Final path parameter: {progress:.1f}, expected > {min_progress:.1f}"
    )

    assert (max_contouring := errors.max_contouring) < max_contouring_error, (
        f"Vehicle deviated too far from the reference trajectory. "
        f"Max contouring error: {max_contouring:.2f} m, expected < {max_contouring_error:.2f} m"
    )

    assert (max_lag := errors.max_lag) < max_lag_error, (
        f"Vehicle had excessive lag error along the reference trajectory. "
        f"Max lag error: {max_lag:.2f} m, expected < {max_lag_error:.2f} m"
    )

    assert len(trajectories) == t + 1, (
        f"Expected {t + 1} collected trajectories, but found {len(trajectories)}."
    )


@mark.parametrize(
    ["configuration", "configuration_name"],
    [
        (
            mpcc.numpy.planner_from_augmented(
                reference=reference.numpy.loop, obstacles=obstacles.numpy.static.loop
            ),
            "numpy-from-augmented-static",
        ),
        (
            mpcc.numpy.planner_from_mpcc(
                reference=reference.numpy.loop, obstacles=obstacles.numpy.static.loop
            ),
            "numpy-from-mpcc-static",
        ),
        (
            mpcc.numpy.planner_from_mpcc(
                reference=reference.numpy.slalom,
                obstacles=obstacles.numpy.dynamic.slalom,
            ),
            "numpy-from-mpcc-dynamic",
        ),
        (
            mpcc.numpy.planner_from_mpcc(
                reference=reference.numpy.short,
                obstacles=obstacles.numpy.dynamic.short,
                use_covariance_propagation=True,
            ),
            "numpy-from-mpcc-dynamic-uncertain",
        ),
        (
            mpcc.jax.planner_from_augmented(
                reference=reference.jax.loop, obstacles=obstacles.jax.static.loop
            ),
            "jax-from-augmented-static",
        ),
        (
            mpcc.jax.planner_from_mpcc(
                reference=reference.jax.loop, obstacles=obstacles.jax.static.loop
            ),
            "jax-from-mpcc-static",
        ),
        (
            mpcc.jax.planner_from_mpcc(
                reference=reference.jax.slalom, obstacles=obstacles.jax.dynamic.slalom
            ),
            "jax-from-mpcc-dynamic",
        ),
        (
            mpcc.jax.planner_from_mpcc(
                reference=reference.jax.short,
                obstacles=obstacles.jax.dynamic.short,
                use_covariance_propagation=True,
            ),
            "jax-from-mpcc-dynamic-uncertain",
        ),
    ],
)
@mark.visualize.with_args(visualizer.mpcc(), lambda seed: f"{seed}-obstacles")
@mark.filterwarnings("error")
@mark.integration
def test_that_mpcc_planner_follows_trajectory_without_collision_when_obstacles_are_present[
    StateT: AugmentedState,
    StateBatchT: StateBatch,
    InputSequenceT: ControlInputSequence,
    InputBatchT: ControlInputBatch,
    ObstacleStatesT: ObstacleStates,
](
    visualization: VisualizationData[MpccSimulationResult],
    configuration: MpccPlannerConfiguration[
        StateT, StateBatchT, InputSequenceT, InputBatchT, ObstacleStatesT
    ],
    configuration_name: str,
) -> None:
    reference = configuration.reference
    planner = configuration.planner
    model = configuration.model
    wheelbase = configuration.wheelbase
    registry = configuration.registry
    error_metric, collision_metric = configuration.metrics
    obstacle_simulator = configuration.obstacle_simulator
    obstacle_state_observer = configuration.obstacle_state_observer

    current_state = configuration.initial_state
    nominal_input = configuration.nominal_input

    assert collision_metric is not None, (
        "Collision metric must be provided for this test."
    )
    assert obstacle_simulator is not None, (
        "Obstacle simulator must be provided for this test."
    )
    assert obstacle_state_observer is not None, (
        "Obstacle state observer must be provided for this test."
    )

    min_progress = reference.path_length * 0.7
    progress = 0.0

    for t in range(max_steps := 350):
        control = planner.step(
            temperature=0.05, nominal_input=nominal_input, initial_state=current_state
        )

        nominal_input = control.nominal
        current_state = model.step(inputs=control.optimal, state=current_state)
        obstacle_state_observer.observe(obstacle_simulator.step())

        if (progress := current_state.virtual.array[0]) >= min_progress:
            break

    trajectories = registry.data(access.trajectories.require())
    errors = registry.get(error_metric)
    collision = registry.get(collision_metric)

    visualization.data_is(
        MpccSimulationResult(
            reference=reference,
            states=registry.data(access.states.require()),
            optimal_trajectories=[it.optimal for it in trajectories],
            nominal_trajectories=[it.nominal for it in trajectories],
            contouring_errors=errors.contouring,
            lag_errors=errors.lag,
            time_step_size=model.time_step_size,
            wheelbase=wheelbase,
            max_contouring_error=(max_contouring_error := 5.0),
            max_lag_error=(max_lag_error := 7.5),
            obstacles=registry.data(access.obstacle_states.require()),
            obstacle_forecasts=registry.data(access.obstacle_forecasts.require()),
            controls=registry.data(access.controls.require()),
            risks=registry.data(access.risks),
        )
    ).seed_is(configuration_name)

    assert progress > min_progress, (
        f"Vehicle did not make sufficient progress along the path in {max_steps} steps. "
        f"Final path parameter: {progress:.1f}, expected > {min_progress:.1f}"
    )

    assert (max_contouring := errors.max_contouring) < max_contouring_error, (
        f"Vehicle deviated too far from the reference trajectory. "
        f"Max lateral deviation: {max_contouring:.2f} m, expected < {max_contouring_error:.2f} m"
    )

    assert (max_lag := errors.max_lag) < max_lag_error, (
        f"Vehicle had excessive lag error along the reference trajectory. "
        f"Max lag error: {max_lag:.2f} m, expected < {max_lag_error:.2f} m"
    )

    assert (min_distance := collision.min_distances.min()) > (safe_distance := 0.0), (
        f"Vehicle came too close to obstacles. "
        f"Min distance to obstacles: {min_distance:.2f} m, expected > {safe_distance:.2f} m"
    )

    assert len(trajectories) == t + 1, (
        f"Expected {t + 1} collected trajectories, but found {len(trajectories)}."
    )
