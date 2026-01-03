from typing import Final, Sequence, Protocol, Self
from dataclasses import dataclass, field

from trajax import (
    Mppi,
    AugmentedModel,
    AugmentedSampler,
    ContouringCost,
    LagCost,
    Trajectory,
    Circles,
    Distance,
    DistanceExtractor,
    ObstacleMotionPredictor,
    RiskCollector,
    ControlCollector,
    mppi,
    model,
    sampler,
    costs,
    trajectory,
    types,
    extract,
    distance,
    obstacles as create_obstacles,
    predictor,
    propagator,
    risk,
)

from numtypes import array, Array, Dim1, Dim2

import numpy as np

from tests.examples.common import SimulatingObstacleStateProvider

HORIZON: Final = 30

type PhysicalState = types.numpy.bicycle.State
type PhysicalStateBatch = types.numpy.bicycle.StateBatch
type PhysicalInputSequence = types.numpy.bicycle.ControlInputSequence
type PhysicalInputBatch = types.numpy.bicycle.ControlInputBatch
type VirtualState = types.numpy.simple.State
type VirtualStateBatch = types.numpy.simple.StateBatch
type VirtualInputSequence = types.numpy.simple.ControlInputSequence
type VirtualInputBatch = types.numpy.simple.ControlInputBatch
type MpccState = types.numpy.augmented.State[PhysicalState, VirtualState]
type MpccStateBatch = types.numpy.augmented.StateBatch[
    PhysicalStateBatch, VirtualStateBatch
]
type MpccInputSequence = types.numpy.augmented.ControlInputSequence[
    PhysicalInputSequence, VirtualInputSequence
]
type MpccInputBatch = types.numpy.augmented.ControlInputBatch[
    PhysicalInputBatch, VirtualInputBatch
]
type Planner = Mppi[MpccState, MpccInputSequence]
type ObstacleStates = types.numpy.ObstacleStates
type ObstacleStatesHistory = types.numpy.ObstacleStatesRunningHistory


class ObstacleStateProvider(SimulatingObstacleStateProvider[ObstacleStates], Protocol):
    def with_time_step(self, time_step: float) -> Self:
        """Returns a new obstacle state provider configured for the given time step size."""
        ...

    def with_predictor(
        self, predictor: ObstacleMotionPredictor[ObstacleStatesHistory, ObstacleStates]
    ) -> Self:
        """Returns a new obstacle state provider using the given motion predictor."""
        ...


def path_parameter(states: VirtualStateBatch) -> types.numpy.PathParameters:
    return types.numpy.path_parameters(states.array[:, 0, :])


def path_velocity(inputs: VirtualInputBatch) -> Array[Dim2]:
    return inputs.array[:, 0, :]


def position(states: PhysicalStateBatch) -> types.numpy.Positions:
    return types.numpy.positions(x=states.positions.x(), y=states.positions.y())


def heading(states: PhysicalStateBatch) -> types.numpy.Headings:
    return types.numpy.headings(theta=states.orientations())


def bicycle_to_obstacle_states(
    states: types.numpy.bicycle.ObstacleStateSequences, covariances: Array | None
) -> ObstacleStates:
    return types.numpy.obstacle_states.create(
        x=states.x(), y=states.y(), heading=states.theta(), covariance=covariances
    )


@dataclass(kw_only=True, frozen=True)
class NumPyMpccPlannerConfiguration:
    reference: Trajectory
    planner: Planner
    model: AugmentedModel
    contouring_cost: ContouringCost
    lag_cost: LagCost
    wheelbase: float

    distance: DistanceExtractor[MpccStateBatch, ObstacleStates, Distance] | None = None
    obstacles: ObstacleStateProvider | None = None
    risk_collector: RiskCollector | None = None
    control_collector: ControlCollector | None = None

    @staticmethod
    def stack_states(states: list[MpccState]) -> MpccStateBatch:
        return types.numpy.augmented.state_batch.of(
            physical=types.numpy.bicycle.state_batch.of_states(
                [it.physical for it in states]
            ),
            virtual=types.numpy.simple.state_batch.of_states(
                [it.virtual for it in states]
            ),
        )

    @staticmethod
    def stack_obstacles(obstacle_states: Sequence[ObstacleStates]) -> ObstacleStates:
        return types.numpy.obstacle_states.of_states(obstacle_states)

    @staticmethod
    def zero_inputs(horizon: int) -> MpccInputBatch:
        return types.numpy.augmented.control_input_batch.of(
            physical=types.numpy.bicycle.control_input_batch.zero(horizon=horizon),
            virtual=types.numpy.simple.control_input_batch.zero(
                horizon=horizon, dimension=1
            ),
        )

    @property
    def initial_state(self) -> MpccState:
        return types.numpy.augmented.state.of(
            physical=types.numpy.bicycle.state(x=0.0, y=0.0, heading=0.0, speed=0.0),
            virtual=types.numpy.simple.state(np.array([0.0])),
        )

    @property
    def nominal_input(self) -> MpccInputSequence:
        return types.numpy.augmented.control_input_sequence.of(
            physical=types.numpy.bicycle.control_input_sequence.zeroes(horizon=HORIZON),
            virtual=types.numpy.simple.control_input_sequence.zeroes(
                horizon=HORIZON, dimension=1
            ),
        )


@dataclass(frozen=True)
class NumPyMpccPlannerWeights:
    contouring: float = 20.0
    lag: float = 10.0
    progress: float = 500.0
    control_smoothing: Array[Dim1] = field(
        default_factory=lambda: array([2.0, 5.0, 5.0], shape=(3,))
    )
    collision: float = 1000.0


@dataclass(frozen=True)
class NumPySamplingOptions:
    physical_standard_deviation: Array[Dim1] = field(
        default_factory=lambda: np.array([1.0, 2.0])
    )
    virtual_standard_deviation: Array[Dim1] = field(
        default_factory=lambda: np.array([1.0])
    )
    rollout_count: int = 512
    physical_seed: int = 42
    virtual_seed: int = 43
    obstacle_seed: int = 44


class reference:
    small_circle: Final = trajectory.numpy.waypoints(
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

    loop: Final = trajectory.numpy.waypoints(
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

    slalom: Final = trajectory.numpy.waypoints(
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

    short: Final = trajectory.numpy.waypoints(
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
    none: Final = create_obstacles.numpy.empty(horizon=HORIZON)

    class static:
        loop: Final = create_obstacles.numpy.static(
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
            horizon=HORIZON,
        )

    class dynamic:
        slalom: Final = create_obstacles.numpy.dynamic(
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
            horizon=HORIZON,
        )

        short: Final = create_obstacles.numpy.dynamic(
            positions=array(
                [[15.0, 10.0]],
                shape=(1, 2),
            ),
            velocities=array(
                [[0.0, -1.5]],
                shape=(1, 2),
            ),
            horizon=HORIZON,
        )


class configure:
    @staticmethod
    def planner_from_base(
        reference: Trajectory = reference.small_circle,
        weights: NumPyMpccPlannerWeights = NumPyMpccPlannerWeights(),
        sampling: NumPySamplingOptions = NumPySamplingOptions(),
    ) -> NumPyMpccPlannerConfiguration:
        # NOTE: Type Checkers like Pyright won't be able to infer complex types, so you may
        # need to help them with an explicit annotation.
        planner: Planner = mppi.numpy.base(
            model=(
                augmented_model := AugmentedModel.of(
                    physical=model.numpy.bicycle.dynamical(
                        time_step_size=(dt := 0.1),
                        wheelbase=(L := 2.5),
                        speed_limits=(0.0, 15.0),
                        steering_limits=(-0.5, 0.5),
                        acceleration_limits=(-3.0, 3.0),
                    ),
                    virtual=model.numpy.integrator.dynamical(
                        time_step_size=dt,
                        state_limits=(0, reference.path_length),
                        velocity_limits=(0, 15),
                    ),
                    state=types.numpy.augmented.state,
                    batch=types.numpy.augmented.state_batch,
                )
            ),
            cost_function=costs.numpy.combined(
                contouring_cost := costs.numpy.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_extractor := extract.from_virtual(path_parameter)
                    ),
                    position_extractor=(
                        position_extractor := extract.from_physical(position)
                    ),
                    weight=weights.contouring,
                ),
                lag_cost := costs.numpy.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=weights.lag,
                ),
                costs.numpy.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(path_velocity),
                    time_step_size=dt,
                    weight=weights.progress,
                ),
                costs.numpy.comfort.control_smoothing(
                    weights=weights.control_smoothing
                ),
            ),
            sampler=AugmentedSampler.of(
                physical=sampler.numpy.gaussian(
                    standard_deviation=sampling.physical_standard_deviation,
                    rollout_count=sampling.rollout_count,
                    to_batch=types.numpy.bicycle.control_input_batch.create,
                    seed=sampling.physical_seed,
                ),
                virtual=sampler.numpy.gaussian(
                    standard_deviation=sampling.virtual_standard_deviation,
                    rollout_count=sampling.rollout_count,
                    to_batch=types.numpy.simple.control_input_batch.create,
                    seed=sampling.virtual_seed,
                ),
                batch=types.numpy.augmented.control_input_batch,
            ),
        )

        return NumPyMpccPlannerConfiguration(
            reference=reference,
            planner=planner,
            model=augmented_model,
            contouring_cost=contouring_cost,
            lag_cost=lag_cost,
            wheelbase=L,
        )

    @staticmethod
    def planner_from_augmented(
        *,
        reference: Trajectory = reference.small_circle,
        obstacles: ObstacleStateProvider = obstacles.none,
        weights: NumPyMpccPlannerWeights = NumPyMpccPlannerWeights(),
        sampling: NumPySamplingOptions = NumPySamplingOptions(),
        use_covariance_propagation: bool = False,
    ) -> NumPyMpccPlannerConfiguration:
        obstacles = obstacles.with_time_step(dt := 0.1).with_predictor(
            predictor.curvilinear(
                horizon=HORIZON,
                model=model.numpy.bicycle.obstacle(
                    time_step_size=dt, wheelbase=(L := 2.5)
                ),
                prediction=bicycle_to_obstacle_states,
                propagator=propagator.numpy.linear(
                    time_step_size=dt,
                    initial_covariance=propagator.numpy.covariance.constant_variance(
                        position_variance=0.01, velocity_variance=1.0
                    ),
                    padding=propagator.padding(to_dimension=3, epsilon=1e-9),
                )
                if use_covariance_propagation
                else None,
            )
        )

        planner, augmented_model = mppi.numpy.augmented(
            models=(
                model.numpy.bicycle.dynamical(
                    time_step_size=dt,
                    wheelbase=L,
                    speed_limits=(0.0, 15.0),
                    steering_limits=(-0.5, 0.5),
                    acceleration_limits=(-3.0, 3.0),
                ),
                model.numpy.integrator.dynamical(
                    time_step_size=dt,
                    state_limits=(0, reference.path_length),
                    velocity_limits=(0, 15),
                ),
            ),
            samplers=(
                sampler.numpy.gaussian(
                    standard_deviation=sampling.physical_standard_deviation,
                    rollout_count=sampling.rollout_count,
                    to_batch=types.numpy.bicycle.control_input_batch.create,
                    seed=sampling.physical_seed,
                ),
                sampler.numpy.gaussian(
                    standard_deviation=sampling.virtual_standard_deviation,
                    rollout_count=sampling.rollout_count,
                    to_batch=types.numpy.simple.control_input_batch.create,
                    seed=sampling.virtual_seed,
                ),
            ),
            cost=costs.numpy.combined(
                contouring_cost := costs.numpy.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_extractor := extract.from_virtual(path_parameter)
                    ),
                    position_extractor=(
                        position_extractor := extract.from_physical(position)
                    ),
                    weight=weights.contouring,
                ),
                lag_cost := costs.numpy.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=weights.lag,
                ),
                costs.numpy.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(path_velocity),
                    time_step_size=dt,
                    weight=weights.progress,
                ),
                costs.numpy.comfort.control_smoothing(
                    weights=weights.control_smoothing
                ),
                costs.numpy.safety.collision(
                    obstacle_states=obstacles,
                    sampler=create_obstacles.sampler.numpy.gaussian(
                        seed=sampling.obstacle_seed
                    ),
                    distance=(
                        circles_distance := distance.numpy.circles(
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
                            risk.collector.decorating(
                                risk.numpy.mean_variance(gamma=0.5, sample_count=10)
                            )
                            if use_covariance_propagation
                            else None
                        )
                    ),
                ),
            ),
            state=types.numpy.augmented.state,
            state_batch=types.numpy.augmented.state_batch,
            input_batch=types.numpy.augmented.control_input_batch,
        )

        planner = (control_collector := mppi.collector.controls.decorating(planner))

        return NumPyMpccPlannerConfiguration(
            reference=reference,
            planner=planner,
            model=augmented_model,
            contouring_cost=contouring_cost,
            lag_cost=lag_cost,
            wheelbase=L,
            distance=circles_distance,
            obstacles=obstacles,
            risk_collector=risk_collector,
            control_collector=control_collector,
        )
