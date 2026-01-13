from typing import Final
from dataclasses import dataclass, field

from trajax import (
    Mppi,
    AugmentedModel,
    DynamicalModel,
    AugmentedSampler,
    Trajectory,
    Circles,
    ObstaclePositionExtractor,
    ObstacleStateObserver,
    ObstacleSimulator,
    MetricRegistry,
    MpccErrorMetric,
    CollisionMetric,
    collectors,
    metrics,
)
from trajax.jax import (
    mppi,
    filters,
    model,
    sampler,
    costs,
    distance,
    boundary,
    trajectory,
    types,
    extract,
    predictor,
    propagator,
    obstacles as create_obstacles,
    risk,
)

from numtypes import array, Array, Dim1
from jaxtyping import Array as JaxArray, Float

import numpy as np
import jax.numpy as jnp
import jax.random as jrandom


type PhysicalState = types.bicycle.State
type PhysicalStateBatch = types.bicycle.StateBatch
type PhysicalInputSequence = types.bicycle.ControlInputSequence
type PhysicalInputBatch = types.bicycle.ControlInputBatch
type VirtualState = types.simple.State
type VirtualStateBatch = types.simple.StateBatch
type VirtualInputSequence = types.simple.ControlInputSequence
type VirtualInputBatch = types.simple.ControlInputBatch
type MpccState = types.augmented.State[PhysicalState, VirtualState]
type MpccStateBatch = types.augmented.StateBatch[PhysicalStateBatch, VirtualStateBatch]
type MpccInputSequence = types.augmented.ControlInputSequence[
    PhysicalInputSequence, VirtualInputSequence
]
type MpccInputBatch = types.augmented.ControlInputBatch[
    PhysicalInputBatch, VirtualInputBatch
]
type Planner = Mppi[MpccState, MpccInputSequence]
type ObstacleStates = types.ObstacleStates
type ObstacleStatesForTimeStep = types.ObstacleStatesForTimeStep
type ObstaclePositions = types.Obstacle2dPositions
type ObstaclePositionsForTimeStep = types.Obstacle2dPositionsForTimeStep


class JaxObstaclePositionExtractor(
    ObstaclePositionExtractor[
        ObstacleStatesForTimeStep,
        ObstacleStates,
        ObstaclePositionsForTimeStep,
        ObstaclePositions,
    ]
):
    def of_states_for_time_step(
        self, states: ObstacleStatesForTimeStep, /
    ) -> ObstaclePositionsForTimeStep:
        return states.positions()

    def of_states(self, states: ObstacleStates, /) -> ObstaclePositions:
        return states.positions()


def path_parameter(states: VirtualStateBatch) -> types.PathParameters:
    return types.path_parameters(states.array[:, 0, :])


def path_velocity(inputs: VirtualInputBatch) -> Float[JaxArray, "T M"]:
    return inputs.array[:, 0, :]


def position(states: PhysicalStateBatch) -> types.Positions:
    return types.positions(x=states.positions.x_array, y=states.positions.y_array)


def heading(states: PhysicalStateBatch) -> types.Headings:
    return types.headings(heading=states.heading_array)


def bicycle_to_obstacle_states(
    states: types.bicycle.ObstacleStateSequences, covariances: JaxArray
) -> ObstacleStates:
    return types.obstacle_states.create(
        x=states.x_array,
        y=states.y_array,
        heading=states.heading_array,
        covariance=covariances,
    )


@dataclass(kw_only=True, frozen=True)
class JaxMpccPlannerConfiguration:
    horizon: int

    reference: Trajectory
    planner: Planner
    model: DynamicalModel
    wheelbase: float
    registry: MetricRegistry
    metrics: tuple[MpccErrorMetric, CollisionMetric | None]

    obstacle_simulator: ObstacleSimulator | None = None
    obstacle_state_observer: ObstacleStateObserver | None = None

    @property
    def initial_state(self) -> MpccState:
        return types.augmented.state.of(
            physical=types.bicycle.state.create(x=0.0, y=0.0, heading=0.0, speed=0.0),
            virtual=types.simple.state.zeroes(dimension=1),
        )

    @property
    def nominal_input(self) -> MpccInputSequence:
        return types.augmented.control_input_sequence.of(
            physical=types.bicycle.control_input_sequence.zeroes(horizon=self.horizon),
            virtual=types.simple.control_input_sequence.zeroes(
                horizon=self.horizon, dimension=1
            ),
        )


@dataclass(frozen=True)
class JaxMpccPlannerWeights:
    contouring: float = 20.0
    lag: float = 10.0
    progress: float = 500.0
    control_smoothing: Array[Dim1] = field(
        default_factory=lambda: array([2.0, 5.0, 5.0], shape=(3,))
    )
    collision: float = 1000.0
    boundary: float = 500.0


@dataclass(frozen=True)
class JaxSamplingOptions:
    physical_standard_deviation: Float[JaxArray, "D_u"] = field(
        default_factory=lambda: jnp.array([1.0, 2.0])
    )
    virtual_standard_deviation: float = 1.0
    rollout_count: int = 512
    physical_key: jrandom.PRNGKey = field(default_factory=lambda: jrandom.PRNGKey(42))
    virtual_key: jrandom.PRNGKey = field(default_factory=lambda: jrandom.PRNGKey(43))
    obstacle_seed: int = 44


class reference:
    small_circle: Final = trajectory.waypoints(
        points=array(
            [
                [0.0, 0.0],
                [3.0, 1.0],
                [4.0, 2.0],
                [5.0, 5.0],
                [4.0, 8.0],
                [3.0, 9.0],
                [0.0, 10.0],
                [-3.0, 9.0],
                [-4.0, 8.0],
                [-5.0, 5.0],
                [-4.0, 2.0],
                [-3.0, 1.0],
                [0.0, 0.0],
            ],
            shape=(13, 2),
        ),
        path_length=30.0,
    )

    short_loop: Final = trajectory.waypoints(
        points=array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [20.0, 5.0],
                [25.0, 15.0],
                [20.0, 25.0],
                [10.0, 25.0],
                [5.0, 15.0],
                [10.0, 5.0],
            ],
            shape=(8, 2),
        ),
        path_length=70.0,
    )

    loop: Final = trajectory.waypoints(
        points=array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [20.0, 5.0],
                [25.0, 15.0],
                [20.0, 25.0],
                [10.0, 25.0],
                [5.0, 15.0],
                [10.0, 5.0],
                [20.0, 0.0],
                [30.0, 0.0],
                [40.0, 5.0],
                [50.0, 5.0],
                [60.0, 0.0],
                [70.0, 0.0],
            ],
            shape=(14, 2),
        ),
        path_length=120.0,
    )

    slalom: Final = trajectory.waypoints(
        points=array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [20.0, 5.0],
                [30.0, 10.0],
                [40.0, 5.0],
                [50.0, 0.0],
                [60.0, 0.0],
                [70.0, 0.0],
            ],
            shape=(8, 2),
        ),
        path_length=100.0,
    )

    short: Final = trajectory.waypoints(
        points=array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [20.0, 5.0],
                [30.0, 5.0],
            ],
            shape=(4, 2),
        ),
        path_length=35.0,
    )


class obstacles:
    none: Final = create_obstacles.empty()

    class static:
        loop: Final = create_obstacles.static(
            positions=array(
                [
                    [15.0, 2.5],
                    [17.0, 20.0],
                    [8.0, 12.0],
                    [12.0, 4.0],
                    [32.0, 0.5],
                    [42.0, 7.0],
                    [57.0, 3.0],
                ],
                shape=(7, 2),
            ),
            headings=array(
                [
                    np.pi / 6,
                    -np.pi / 4,
                    np.pi / 3,
                    0.0,
                    np.pi / 8,
                    -np.pi / 6,
                    np.pi / 2,
                ],
                shape=(7,),
            ),
        )

    class dynamic:
        slalom: Final = create_obstacles.dynamic(
            positions=array(
                [
                    [25.0, 22.5],
                    [55.0, 0.0],
                    [15.0, -5.0],
                    [42.0, 4.0],
                ],
                shape=(4, 2),
            ),
            velocities=array(
                [
                    [0.0, -1.5],
                    [-2.5, 0.0],
                    [0.5, 1.5],
                    [0.0, 0.0],
                ],
                shape=(4, 2),
            ),
        )

        short: Final = create_obstacles.dynamic(
            positions=array(
                [[15.0, 10.0]],
                shape=(1, 2),
            ),
            velocities=array(
                [[0.0, -1.5]],
                shape=(1, 2),
            ),
        )


class configure:
    @staticmethod
    def planner_from_base(
        *,
        horizon: int = 30,
        reference: Trajectory = reference.small_circle,
        weights: JaxMpccPlannerWeights = JaxMpccPlannerWeights(),
        sampling: JaxSamplingOptions = JaxSamplingOptions(),
    ) -> JaxMpccPlannerConfiguration:
        # NOTE: Type Checkers like Pyright won't be able to infer complex types, so you may
        # need to help them with an explicit annotation.
        planner: Planner = mppi.base(
            model=(
                augmented_model := AugmentedModel.of(
                    physical=model.bicycle.dynamical(
                        time_step_size=(dt := 0.1),
                        wheelbase=(L := 2.5),
                        speed_limits=(0.0, 15.0),
                        steering_limits=(-0.5, 0.5),
                        acceleration_limits=(-3.0, 3.0),
                    ),
                    virtual=model.integrator.dynamical(
                        time_step_size=dt,
                        state_limits=(0, reference.path_length),
                        velocity_limits=(0, 15),
                    ),
                    state=types.augmented.state,
                    sequence=types.augmented.state_sequence,
                    batch=types.augmented.state_batch,
                )
            ),
            cost_function=costs.combined(
                contouring_cost := costs.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_extractor := extract.from_virtual(path_parameter)
                    ),
                    position_extractor=(
                        position_extractor := extract.from_physical(position)
                    ),
                    weight=weights.contouring,
                ),
                lag_cost := costs.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=weights.lag,
                ),
                costs.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(path_velocity),
                    time_step_size=dt,
                    weight=weights.progress,
                ),
                costs.comfort.control_smoothing(weights=weights.control_smoothing),
            ),
            sampler=AugmentedSampler.of(
                physical=sampler.gaussian(
                    standard_deviation=sampling.physical_standard_deviation,
                    rollout_count=sampling.rollout_count,
                    to_batch=types.bicycle.control_input_batch.create,
                    key=sampling.physical_key,
                ),
                virtual=sampler.gaussian(
                    standard_deviation=jnp.array([sampling.virtual_standard_deviation]),
                    rollout_count=sampling.rollout_count,
                    to_batch=types.simple.control_input_batch.create,
                    key=sampling.virtual_key,
                ),
                batch=types.augmented.control_input_batch,
            ),
            filter_function=filters.savgol(window_length=11, polynomial_order=3),
        )

        planner = (
            trajectories_collector := collectors.trajectories.decorating(
                control_collector := collectors.controls.decorating(
                    state_collector := collectors.states.decorating(
                        planner,
                        transformer=types.augmented.state_sequence.of_states(
                            physical=types.bicycle.state_sequence.of_states,
                            virtual=types.simple.state_sequence.of_states,
                        ),
                    )
                ),
                model=augmented_model,
            )
        )

        return JaxMpccPlannerConfiguration(
            horizon=horizon,
            reference=reference,
            planner=planner,
            model=augmented_model,
            wheelbase=L,
            registry=metrics.registry(
                mpcc_error_metrics := metrics.mpcc_error(
                    contouring=contouring_cost, lag=lag_cost
                ),
                collectors=collectors.registry(
                    states=state_collector,
                    controls=control_collector,
                    trajectories=trajectories_collector,
                ),
            ),
            metrics=(mpcc_error_metrics, None),
        )

    @staticmethod
    def planner_from_augmented(
        *,
        horizon: int = 30,
        reference: Trajectory = reference.small_circle,
        obstacles: ObstacleSimulator = obstacles.none,
        weights: JaxMpccPlannerWeights = JaxMpccPlannerWeights(),
        sampling: JaxSamplingOptions = JaxSamplingOptions(),
        use_covariance_propagation: bool = False,
    ) -> JaxMpccPlannerConfiguration:
        planner, augmented_model = mppi.augmented(
            models=(
                model.bicycle.dynamical(
                    time_step_size=(dt := 0.1),
                    wheelbase=(L := 2.5),
                    speed_limits=(0.0, 15.0),
                    steering_limits=(-0.5, 0.5),
                    acceleration_limits=(-3.0, 3.0),
                ),
                model.integrator.dynamical(
                    time_step_size=dt,
                    state_limits=(0, reference.path_length),
                    velocity_limits=(0, 15),
                ),
            ),
            samplers=(
                sampler.gaussian(
                    standard_deviation=sampling.physical_standard_deviation,
                    rollout_count=sampling.rollout_count,
                    to_batch=types.bicycle.control_input_batch.create,
                    key=sampling.physical_key,
                ),
                sampler.gaussian(
                    standard_deviation=jnp.array([sampling.virtual_standard_deviation]),
                    rollout_count=sampling.rollout_count,
                    to_batch=types.simple.control_input_batch.create,
                    key=sampling.virtual_key,
                ),
            ),
            cost=costs.combined(
                contouring_cost := costs.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_extractor := extract.from_virtual(path_parameter)
                    ),
                    position_extractor=(
                        position_extractor := extract.from_physical(position)
                    ),
                    weight=weights.contouring,
                ),
                lag_cost := costs.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=weights.lag,
                ),
                costs.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(path_velocity),
                    time_step_size=dt,
                    weight=weights.progress,
                ),
                costs.comfort.control_smoothing(weights=weights.control_smoothing),
                costs.safety.collision(
                    obstacle_states=(
                        forecasts_collector := collectors.obstacle_forecasts.decorating(
                            obstacles_provider := create_obstacles.provider.predicting(
                                predictor=predictor.curvilinear(
                                    horizon=horizon,
                                    model=model.bicycle.obstacle(
                                        time_step_size=dt, wheelbase=L
                                    ),
                                    prediction=bicycle_to_obstacle_states,
                                    propagator=propagator.linear(
                                        time_step_size=dt,
                                        initial_covariance=propagator.covariance.constant_variance(
                                            position_variance=0.01,
                                            velocity_variance=1.0,
                                        ),
                                        padding=propagator.padding(
                                            to_dimension=3, epsilon=1e-9
                                        ),
                                    )
                                    if use_covariance_propagation
                                    else None,
                                ),
                                history=types.obstacle_states_running_history.empty(
                                    horizon=2, obstacle_count=obstacles.obstacle_count
                                ),
                                id_assignment=create_obstacles.id_assignment.hungarian(
                                    position_extractor=JaxObstaclePositionExtractor(),
                                    cutoff=10.0,
                                ),
                            )
                        )
                    ),
                    sampler=create_obstacles.sampler.gaussian(
                        seed=sampling.obstacle_seed
                    ),
                    distance=(
                        circles_distance := distance.circles(
                            ego=Circles(
                                origins=array(
                                    [[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]],
                                    shape=(V := 3, 2),
                                ),
                                radii=array([0.8, 0.8, 0.8], shape=(V,)),
                            ),
                            obstacle=Circles(
                                origins=array(
                                    [[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]],
                                    shape=(C := 3, 2),
                                ),
                                radii=array([0.8, 0.8, 0.8], shape=(C,)),
                            ),
                            position_extractor=position_extractor,
                            heading_extractor=extract.from_physical(heading),
                        )
                    ),
                    distance_threshold=array([0.5, 0.5, 0.5], shape=(V,)),
                    weight=weights.collision,
                    metric=(
                        risk_collector := (
                            collectors.risk.decorating(
                                risk.mean_variance(gamma=0.5, sample_count=10)
                            )
                            if use_covariance_propagation
                            else None
                        )
                    ),
                ),
            ),
            state=types.augmented.state,
            state_sequence=types.augmented.state_sequence,
            state_batch=types.augmented.state_batch,
            input_batch=types.augmented.control_input_batch,
            filter_function=filters.savgol(window_length=11, polynomial_order=3),
        )

        planner = (
            trajectories_collector := collectors.trajectories.decorating(
                control_collector := collectors.controls.decorating(
                    state_collector := collectors.states.decorating(
                        planner,
                        transformer=types.augmented.state_sequence.of_states(
                            physical=types.bicycle.state_sequence.of_states,
                            virtual=types.simple.state_sequence.of_states,
                        ),
                    )
                ),
                model=augmented_model,
            )
        )

        obstacle_collector = collectors.obstacle_states.decorating(
            obstacles_provider, transformer=types.obstacle_states.of_states
        )

        return JaxMpccPlannerConfiguration(
            horizon=horizon,
            reference=reference,
            planner=planner,
            model=augmented_model,
            wheelbase=L,
            registry=metrics.registry(
                mpcc_error_metrics := metrics.mpcc_error(
                    contouring=contouring_cost, lag=lag_cost
                ),
                collision_metrics := metrics.collision(
                    distance_threshold=0.0, distance=circles_distance
                ),
                collectors=collectors.registry(
                    states=state_collector,
                    controls=control_collector,
                    risks=risk_collector,
                    trajectories=trajectories_collector,
                    obstacles=obstacle_collector,
                    obstacle_forecasts=forecasts_collector,
                ),
            ),
            metrics=(mpcc_error_metrics, collision_metrics),
            obstacle_simulator=obstacles.with_time_step_size(dt),
            obstacle_state_observer=obstacle_collector,
        )

    @staticmethod
    def planner_from_mpcc(
        *,
        horizon: int = 30,
        reference: Trajectory = reference.small_circle,
        obstacles: ObstacleSimulator = obstacles.none,
        weights: JaxMpccPlannerWeights = JaxMpccPlannerWeights(),
        sampling: JaxSamplingOptions = JaxSamplingOptions(),
        use_covariance_propagation: bool = False,
        use_boundary: bool = False,
    ) -> JaxMpccPlannerConfiguration:
        planner, augmented_model, contouring_cost, lag_cost = mppi.mpcc(
            model=model.bicycle.dynamical(
                time_step_size=(dt := 0.1),
                wheelbase=(L := 2.5),
                speed_limits=(0.0, 15.0),
                steering_limits=(-0.5, 0.5),
                acceleration_limits=(-3.0, 3.0),
            ),
            sampler=sampler.gaussian(
                standard_deviation=sampling.physical_standard_deviation,
                rollout_count=sampling.rollout_count,
                to_batch=types.bicycle.control_input_batch.create,
                key=sampling.physical_key,
            ),
            costs=(
                costs.comfort.control_smoothing(weights=weights.control_smoothing),
                costs.safety.collision(
                    obstacle_states=(
                        forecasts_collector := collectors.obstacle_forecasts.decorating(
                            obstacles_provider := create_obstacles.provider.predicting(
                                predictor=predictor.curvilinear(
                                    horizon=horizon,
                                    model=model.bicycle.obstacle(
                                        time_step_size=dt, wheelbase=L
                                    ),
                                    prediction=bicycle_to_obstacle_states,
                                    propagator=propagator.linear(
                                        time_step_size=dt,
                                        initial_covariance=propagator.covariance.constant_variance(
                                            position_variance=0.01,
                                            velocity_variance=1.0,
                                        ),
                                        padding=propagator.padding(
                                            to_dimension=3, epsilon=1e-9
                                        ),
                                    )
                                    if use_covariance_propagation
                                    else None,
                                ),
                                history=types.obstacle_states_running_history.empty(
                                    horizon=2, obstacle_count=obstacles.obstacle_count
                                ),
                                id_assignment=create_obstacles.id_assignment.hungarian(
                                    position_extractor=JaxObstaclePositionExtractor(),
                                    cutoff=10.0,
                                ),
                            )
                        )
                    ),
                    sampler=create_obstacles.sampler.gaussian(
                        seed=sampling.obstacle_seed
                    ),
                    distance=(
                        circles_distance := distance.circles(
                            ego=Circles(
                                origins=array(
                                    [[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]],
                                    shape=(V := 3, 2),
                                ),
                                radii=array([0.8, 0.8, 0.8], shape=(V,)),
                            ),
                            obstacle=Circles(
                                origins=array(
                                    [[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]],
                                    shape=(C := 3, 2),
                                ),
                                radii=array([0.8, 0.8, 0.8], shape=(C,)),
                            ),
                            position_extractor=(
                                position_extractor := extract.from_physical(position)
                            ),
                            heading_extractor=extract.from_physical(heading),
                        )
                    ),
                    distance_threshold=array([0.5, 0.5, 0.5], shape=(V,)),
                    weight=weights.collision,
                    metric=(
                        risk_collector := (
                            collectors.risk.decorating(
                                risk.mean_variance(gamma=0.5, sample_count=10)
                            )
                            if use_covariance_propagation
                            else None
                        )
                    ),
                ),
            )
            + (
                (
                    costs.safety.boundary(
                        distance=boundary.fixed_width(
                            reference=reference,
                            position_extractor=position_extractor,
                            left=3.0,
                            right=3.0,
                        ),
                        distance_threshold=0.25,
                        weight=weights.boundary,
                    ),
                )
                if use_boundary
                else ()
            ),
            reference=reference,
            position_extractor=position_extractor,
            config={
                "weights": {
                    "contouring": weights.contouring,
                    "lag": weights.lag,
                    "progress": weights.progress,
                },
                "virtual": {
                    "velocity_limits": (0.0, 15.0),
                    "sampling_standard_deviation": sampling.virtual_standard_deviation,
                    "sampling_key": sampling.virtual_key,
                },
            },
            filter_function=filters.savgol(window_length=11, polynomial_order=3),
        )

        planner = (
            trajectories_collector := collectors.trajectories.decorating(
                control_collector := collectors.controls.decorating(
                    state_collector := collectors.states.decorating(
                        planner,
                        transformer=types.augmented.state_sequence.of_states(
                            physical=types.bicycle.state_sequence.of_states,
                            virtual=types.simple.state_sequence.of_states,
                        ),
                    )
                ),
                model=augmented_model,
            )
        )

        obstacle_collector = collectors.obstacle_states.decorating(
            obstacles_provider, transformer=types.obstacle_states.of_states
        )

        return JaxMpccPlannerConfiguration(
            horizon=horizon,
            reference=reference,
            planner=planner,
            model=augmented_model,
            wheelbase=L,
            registry=metrics.registry(
                mpcc_error_metrics := metrics.mpcc_error(
                    contouring=contouring_cost, lag=lag_cost
                ),
                collision_metrics := metrics.collision(
                    distance_threshold=0.0, distance=circles_distance
                ),
                collectors=collectors.registry(
                    states=state_collector,
                    controls=control_collector,
                    risks=risk_collector,
                    trajectories=trajectories_collector,
                    obstacles=obstacle_collector,
                    obstacle_forecasts=forecasts_collector,
                ),
            ),
            metrics=(mpcc_error_metrics, collision_metrics),
            obstacle_simulator=obstacles.with_time_step_size(dt),
            obstacle_state_observer=obstacle_collector,
        )
