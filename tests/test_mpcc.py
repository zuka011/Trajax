from typing import Protocol, Sequence

from trajax import (
    AugmentedState,
    StateBatch,
    ControlInputSequence,
    ControlInputBatch,
    DynamicalModel,
    Trajectory,
    ContouringCost,
    LagCost,
    Distance,
    DistanceExtractor,
    ObstacleStates,
    SampledObstacleStates,
    Weights,
    Mppi,
    RiskCollector,
    ControlCollector,
)

import numpy as np
from numtypes import Array, Dim1

from tests.visualize import VisualizationData, visualizer, MpccSimulationResult
from tests.examples import mpcc, reference, obstacles, SimulatingObstacleStateProvider
from pytest import mark


class StateStacker[StateT, StateBatchT](Protocol):
    def __call__(self, states: Sequence[StateT]) -> StateBatchT:
        """Stacks a sequence of states into a state batch."""
        ...


class ZeroControlInputProvider[ControlInputBatchT](Protocol):
    def __call__(self, horizon: int) -> ControlInputBatchT:
        """Returns a control input batch of zeroes with the specified horizon."""
        ...


class ObstacleStacker[ObstacleStatesT](Protocol):
    def __call__(self, obstacle_states: Sequence[ObstacleStatesT]) -> ObstacleStatesT:
        """Stacks a sequence of obstacle states into a single obstacle states batch."""
        ...


class MpccPlannerConfiguration[
    StateT: AugmentedState,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
    ObstacleStatesT: ObstacleStates,
](Protocol):
    @property
    def reference(self) -> Trajectory:
        """Returns the reference trajectory."""
        ...

    @property
    def obstacles(self) -> SimulatingObstacleStateProvider[ObstacleStatesT] | None:
        """Returns the obstacle state provider."""
        ...

    @property
    def planner(self) -> Mppi[StateT, ControlInputSequenceT, Weights]:
        """Returns the MPCC planner."""
        ...

    @property
    def model(
        self,
    ) -> DynamicalModel[StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT]:
        """Returns the augmented dynamical model."""
        ...

    @property
    def contouring_cost(self) -> ContouringCost[ControlInputBatchT, StateBatchT]:
        """Returns the contouring cost function."""
        ...

    @property
    def lag_cost(self) -> LagCost[ControlInputBatchT, StateBatchT]:
        """Returns the lag cost function."""
        ...

    @property
    def distance(
        self,
    ) -> DistanceExtractor[StateBatchT, ObstacleStatesT, Distance] | None:
        """Returns the distance extractor used for obstacle avoidance."""
        ...

    @property
    def risk_collector(self) -> RiskCollector | None:
        """Returns the risk collector used by the planner."""
        ...

    @property
    def control_collector(self) -> ControlCollector | None:
        """Returns the control collector used by the planner."""
        ...

    @property
    def initial_state(self) -> StateT:
        """Returns the initial augmented state."""
        ...

    @property
    def nominal_input(self) -> ControlInputSequenceT:
        """Returns the nominal control input sequence."""
        ...

    @property
    def wheelbase(self) -> float:
        """Returns the vehicle wheelbase (to be used in the visualization)."""
        ...

    @property
    def stack_states(self) -> StateStacker[StateT, StateBatchT]:
        """Returns the state stacker."""
        ...

    @property
    def stack_obstacles(self) -> ObstacleStacker[ObstacleStatesT]:
        """Returns the obstacle stacker."""
        ...

    @property
    def zero_inputs(self) -> ZeroControlInputProvider[ControlInputBatchT]:
        """Returns the control input provider."""
        ...


@mark.parametrize(
    ["configuration", "configuration_name"],
    [
        (mpcc.numpy.planner_from_base(), "numpy-from-base"),
        (mpcc.numpy.planner_from_augmented(), "numpy-from-augmented"),
        (mpcc.jax.planner_from_base(), "jax-from-base"),
        (mpcc.jax.planner_from_augmented(), "jax-from-augmented"),
    ],
)
@mark.visualize.with_args(visualizer.mpcc(), lambda seed: seed)
@mark.integration
def test_that_mpcc_planner_follows_trajectory_without_excessive_deviation[
    StateT: AugmentedState,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
    ObstacleStatesT: ObstacleStates,
](
    visualization: VisualizationData[MpccSimulationResult],
    configuration: MpccPlannerConfiguration[
        StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT, ObstacleStatesT
    ],
    configuration_name: str,
) -> None:
    reference = configuration.reference
    planner = configuration.planner
    augmented_model = configuration.model
    contouring_cost = configuration.contouring_cost
    lag_cost = configuration.lag_cost
    current_state = configuration.initial_state
    nominal_input = configuration.nominal_input
    L = configuration.wheelbase

    states: list[StateT] = []
    min_progress = reference.path_length * 0.9
    progress = 0.0

    for _ in range(max_steps := 150):
        control = planner.step(
            temperature=0.05,
            nominal_input=nominal_input,
            initial_state=current_state,
        )

        nominal_input = control.nominal

        states.append(
            current_state := augmented_model.step(
                input=control.optimal, state=current_state
            )
        )

        if (progress := current_state.virtual.array[0]) >= min_progress:
            break

    visualization.data_is(
        MpccSimulationResult(
            reference=reference,
            states=states,
            contouring_errors=(
                errors_c := contouring_errors(
                    contouring_cost,
                    inputs=(
                        zero_inputs := configuration.zero_inputs(horizon=len(states))
                    ),
                    states=(stacked_states := configuration.stack_states(states)),
                )
            ),
            lag_errors=(
                errors_l := lag_errors(
                    lag_cost, inputs=zero_inputs, states=stacked_states
                )
            ),
            wheelbase=L,
            max_contouring_error=(max_contouring_error := 1.5),
            max_lag_error=(max_lag_error := 5.0),
        )
    ).seed_is(configuration_name)

    assert progress > min_progress, (
        f"Vehicle did not make sufficient progress along the path in {max_steps} steps. "
        f"Final path parameter: {progress:.1f}, expected > {min_progress:.1f}"
    )

    assert (deviation := np.abs(errors_c).max()) < max_contouring_error, (
        f"Vehicle deviated too far from the reference trajectory. "
        f"Max lateral deviation: {deviation:.2f} m, expected < {max_contouring_error:.2f} m"
    )

    assert (max_lag := np.abs(errors_l).max()) < max_lag_error, (
        f"Vehicle had excessive lag error along the reference trajectory. "
        f"Max lag error: {max_lag:.2f} m, expected < {max_lag_error:.2f} m"
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
            mpcc.numpy.planner_from_augmented(
                reference=reference.numpy.slalom,
                obstacles=obstacles.numpy.dynamic.slalom,
            ),
            "numpy-from-augmented-dynamic",
        ),
        (
            mpcc.numpy.planner_from_augmented(
                reference=reference.numpy.short,
                obstacles=obstacles.numpy.dynamic.short,
                use_covariance_propagation=True,
            ),
            "numpy-from-augmented-dynamic-uncertain",
        ),
        (
            mpcc.jax.planner_from_augmented(
                reference=reference.jax.loop, obstacles=obstacles.jax.static.loop
            ),
            "jax-from-augmented-static",
        ),
        (
            mpcc.jax.planner_from_augmented(
                reference=reference.jax.slalom, obstacles=obstacles.jax.dynamic.slalom
            ),
            "jax-from-augmented-dynamic",
        ),
        (
            mpcc.jax.planner_from_augmented(
                reference=reference.jax.short,
                obstacles=obstacles.jax.dynamic.short,
                use_covariance_propagation=True,
            ),
            "jax-from-augmented-dynamic-uncertain",
        ),
    ],
)
@mark.visualize.with_args(visualizer.mpcc(), lambda seed: f"{seed}-obstacles")
@mark.integration
def test_that_mpcc_planner_follows_trajectory_without_collision_when_obstacles_are_present[
    StateT: AugmentedState,
    StateBatchT: StateBatch,
    ControlInputSequenceT: ControlInputSequence,
    ControlInputBatchT: ControlInputBatch,
    ObstacleStatesT: ObstacleStates,
](
    visualization: VisualizationData[MpccSimulationResult],
    configuration: MpccPlannerConfiguration[
        StateT, StateBatchT, ControlInputSequenceT, ControlInputBatchT, ObstacleStatesT
    ],
    configuration_name: str,
) -> None:
    reference = configuration.reference
    planner = configuration.planner
    augmented_model = configuration.model
    contouring_cost = configuration.contouring_cost
    lag_cost = configuration.lag_cost
    current_state = configuration.initial_state
    nominal_input = configuration.nominal_input
    risk_collector = configuration.risk_collector
    control_collector = configuration.control_collector
    L = configuration.wheelbase

    assert (obstacles := configuration.obstacles) is not None, (
        "Obstacles must be provided for this test."
    )
    assert (distance := configuration.distance) is not None, (
        "A distance extractor must be provided for this test."
    )

    states: list[StateT] = []
    obstacle_history: list[ObstacleStatesT] = []
    min_progress = reference.path_length * 0.7
    progress = 0.0

    for _ in range(max_steps := 350):
        control = planner.step(
            temperature=0.05,
            nominal_input=nominal_input,
            initial_state=current_state,
        )

        nominal_input = control.nominal

        states.append(
            current_state := augmented_model.step(
                input=control.optimal, state=current_state
            )
        )

        obstacle_history.append(obstacles())
        obstacles.step()

        if (progress := current_state.virtual.array[0]) >= min_progress:
            break

    visualization.data_is(
        MpccSimulationResult(
            reference=reference,
            states=states,
            contouring_errors=(
                errors_c := contouring_errors(
                    contouring_cost,
                    inputs=(
                        zero_inputs := configuration.zero_inputs(horizon=len(states))
                    ),
                    states=(stacked_states := configuration.stack_states(states)),
                )
            ),
            lag_errors=(
                errors_l := lag_errors(
                    lag_cost, inputs=zero_inputs, states=stacked_states
                )
            ),
            wheelbase=L,
            max_contouring_error=(max_contouring_error := 5.0),
            max_lag_error=(max_lag_error := 7.5),
            obstacles=obstacle_history,
            controls=(
                control_collector.collected if control_collector is not None else ()
            ),
            risks=risk_collector.collected if risk_collector is not None else (),
        )
    ).seed_is(configuration_name)

    assert progress > min_progress, (
        f"Vehicle did not make sufficient progress along the path in {max_steps} steps. "
        f"Final path parameter: {progress:.1f}, expected > {min_progress:.1f}"
    )

    assert (deviation := np.abs(errors_c).max()) < max_contouring_error, (
        f"Vehicle deviated too far from the reference trajectory. "
        f"Max lateral deviation: {deviation:.2f} m, expected < {max_contouring_error:.2f} m"
    )

    assert (max_lag := np.abs(errors_l).max()) < max_lag_error, (
        f"Vehicle had excessive lag error along the reference trajectory. "
        f"Max lag error: {max_lag:.2f} m, expected < {max_lag_error:.2f} m"
    )

    assert (
        min_distance := min_distance_to_obstacles(
            distance,
            states=stacked_states,
            obstacle_states=configuration.stack_obstacles(obstacle_history),
        )
    ) > (safe_distance := 0.0), (
        f"Vehicle came too close to obstacles. "
        f"Min distance to obstacles: {min_distance:.2f} m, expected > {safe_distance:.2f} m"
    )


def contouring_errors[InputT: ControlInputBatch, StateT: StateBatch](
    contouring: ContouringCost[InputT, StateT], inputs: InputT, states: StateT
) -> Array[Dim1]:
    return np.asarray(contouring.error(inputs=inputs, states=states))[:, 0]


def lag_errors[InputT: ControlInputBatch, StateT: StateBatch](
    lag: LagCost[InputT, StateT], inputs: InputT, states: StateT
) -> Array[Dim1]:
    return np.asarray(lag.error(inputs=inputs, states=states))[:, 0]


def min_distance_to_obstacles[
    StateT: StateBatch,
    ObstacleStatesT: ObstacleStates,
    SampleT: SampledObstacleStates,
](
    distance_extractor: DistanceExtractor[StateT, ObstacleStatesT, Distance],
    states: StateT,
    obstacle_states: ObstacleStatesT,
) -> float:
    return np.min(
        distance_extractor(states=states, obstacle_states=obstacle_states.single())
    )
