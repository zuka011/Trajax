from typing import Final, Sequence, Protocol, Self
from dataclasses import dataclass, field

from trajax import (
    Mppi,
    AugmentedModel,
    AugmentedSampler,
    Circles,
    ContouringCost,
    Distance,
    DistanceExtractor,
    Trajectory,
    ObstacleMotionPredictor,
    mppi,
    model,
    sampler,
    costs,
    distance,
    trajectory,
    types,
    classes,
    extract,
    predictor,
    propagator,
    obstacles as create_obstacles,
    risk,
)

from numtypes import array, Array, Dim1
from jaxtyping import Array as JaxArray, Float

import jax.numpy as jnp
import jax.random as jrandom

HORIZON: Final = 30

type PhysicalState = types.jax.bicycle.State
type PhysicalStateBatch = types.jax.bicycle.StateBatch
type PhysicalInputSequence = types.jax.bicycle.ControlInputSequence
type PhysicalInputBatch = types.jax.bicycle.ControlInputBatch
type VirtualState = types.jax.simple.State
type VirtualStateBatch = types.jax.simple.StateBatch
type VirtualInputSequence = types.jax.simple.ControlInputSequence
type VirtualInputBatch = types.jax.simple.ControlInputBatch
type MpccState = types.jax.augmented.State[PhysicalState, VirtualState]
type MpccStateBatch = types.jax.augmented.StateBatch[
    PhysicalStateBatch, VirtualStateBatch
]
type MpccInputSequence = types.jax.augmented.ControlInputSequence[
    PhysicalInputSequence, VirtualInputSequence
]
type MpccInputBatch = types.jax.augmented.ControlInputBatch[
    PhysicalInputBatch, VirtualInputBatch
]
type Model = AugmentedModel[
    PhysicalState,
    PhysicalStateBatch,
    PhysicalInputSequence,
    PhysicalInputBatch,
    VirtualState,
    VirtualStateBatch,
    VirtualInputSequence,
    VirtualInputBatch,
    MpccState,
    MpccStateBatch,
]
type Sampler = AugmentedSampler[
    PhysicalInputSequence,
    PhysicalInputBatch,
    VirtualInputSequence,
    VirtualInputBatch,
    MpccInputBatch,
]
type Planner = Mppi[MpccState, MpccInputSequence]
type ObstacleStates = types.jax.ObstacleStates
type ObstacleStatesHistory = types.jax.ObstacleStatesRunningHistory


class ObstacleStateProvider(
    classes.jax.ObstacleStateProvider[ObstacleStates], Protocol
):
    def with_time_step(self, time_step: float) -> Self:
        """Returns a new obstacle state provider configured with the given time step size."""
        ...

    def with_predictor(
        self, predictor: ObstacleMotionPredictor[ObstacleStatesHistory, ObstacleStates]
    ) -> Self:
        """Returns a new obstacle state provider configured with the given motion predictor."""
        ...

    def step(self) -> None:
        """Advances the internal state of the obstacle state provider."""
        ...


def path_parameter(states: VirtualStateBatch) -> types.jax.PathParameters:
    return types.jax.path_parameters(states.array[:, 0, :])


def path_velocity(inputs: VirtualInputBatch) -> Float[JaxArray, "T M"]:
    return inputs.array[:, 0, :]


def position(states: PhysicalStateBatch) -> types.jax.Positions:
    return types.jax.positions(x=states.positions.x_array, y=states.positions.y_array)


def heading(states: PhysicalStateBatch) -> types.jax.Headings:
    return types.jax.headings(theta=states.orientations_array)


def bicycle_to_obstacle_states(
    states: types.jax.bicycle.ObstacleStateSequences, covariances: JaxArray
) -> ObstacleStates:
    return types.jax.obstacle_states.create(
        x=states.x_array,
        y=states.y_array,
        heading=states.theta_array,
        covariance=covariances,
    )


@dataclass(kw_only=True, frozen=True)
class JaxMpccPlannerConfiguration:
    reference: Trajectory
    planner: Planner
    model: AugmentedModel
    contouring_cost: ContouringCost
    wheelbase: float

    distance: DistanceExtractor[MpccStateBatch, ObstacleStates, Distance] | None = None
    obstacles: ObstacleStateProvider | None = None

    @staticmethod
    def stack_states(states: Sequence[MpccState]) -> MpccStateBatch:
        return types.jax.augmented.state_batch.of(
            physical=types.jax.bicycle.state_batch.of_states(
                [it.physical for it in states]
            ),
            virtual=types.jax.simple.state_batch.of_states(
                [it.virtual for it in states]
            ),
        )

    @staticmethod
    def stack_obstacles(obstacle_states: Sequence[ObstacleStates]) -> ObstacleStates:
        return types.jax.obstacle_states.of_states(obstacle_states)

    @staticmethod
    def zero_inputs(horizon: int) -> MpccInputBatch:
        return types.jax.augmented.control_input_batch.of(
            physical=types.jax.bicycle.control_input_batch.zero(horizon=horizon),
            virtual=types.jax.simple.control_input_batch.zero(
                horizon=horizon, dimension=1
            ),
        )

    @property
    def initial_state(self) -> MpccState:
        return types.jax.augmented.state.of(
            physical=types.jax.bicycle.state(jnp.array([0.0, 0.0, 0.0, 0.0])),
            virtual=types.jax.simple.state(jnp.array([0.0])),
        )

    @property
    def nominal_input(self) -> MpccInputSequence:
        return types.jax.augmented.control_input_sequence.of(
            physical=types.jax.bicycle.control_input_sequence.zeroes(
                horizon=(horizon := 30)
            ),
            virtual=types.jax.simple.control_input_sequence.zeroes(
                horizon=horizon, dimension=1
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


@dataclass(frozen=True)
class JaxSamplingOptions:
    physical_standard_deviation: Float[JaxArray, "D_u"] = field(
        default_factory=lambda: jnp.array([1.0, 2.0])
    )
    virtual_standard_deviation: Float[JaxArray, "D_v"] = field(
        default_factory=lambda: jnp.array([1.0])
    )
    rollout_count: int = 512
    physical_key: jrandom.PRNGKey = field(default_factory=lambda: jrandom.PRNGKey(42))
    virtual_key: jrandom.PRNGKey = field(default_factory=lambda: jrandom.PRNGKey(43))


class reference:
    small_circle: Final = trajectory.jax.waypoints(
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

    loop: Final = trajectory.jax.waypoints(
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

    slalom: Final = trajectory.jax.waypoints(
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

    short: Final = trajectory.jax.waypoints(
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
    none: Final = create_obstacles.jax.empty(horizon=HORIZON)

    class static:
        loop: Final = create_obstacles.jax.static(
            positions=jnp.array(
                [
                    [15.0, 2.5],
                    [17.0, 20.0],
                    [8.0, 12.0],
                    [12.0, 4.0],
                    [32.0, 0.5],
                    [42.0, 7.0],
                    [57.0, 3.0],
                ]
            ),
            headings=jnp.array(
                [
                    jnp.pi / 6,
                    -jnp.pi / 4,
                    jnp.pi / 3,
                    0.0,
                    jnp.pi / 8,
                    -jnp.pi / 6,
                    jnp.pi / 2,
                ]
            ),
            horizon=HORIZON,
        )

    class dynamic:
        slalom: Final = create_obstacles.jax.dynamic(
            positions=jnp.array(
                [
                    [25.0, 22.5],
                    [55.0, 0.0],
                    [15.0, -5.0],
                    [42.0, 4.0],
                ]
            ),
            velocities=jnp.array(
                [
                    [0.0, -1.5],
                    [-2.5, 0.0],
                    [0.5, 1.5],
                    [0.0, 0.0],
                ]
            ),
            horizon=HORIZON,
        )

        short: Final = create_obstacles.jax.dynamic(
            positions=jnp.array(
                [
                    [15.0, 10.0],
                ]
            ),
            velocities=jnp.array(
                [
                    [0.0, -1.5],
                ]
            ),
            horizon=HORIZON,
        )


class configure:
    @staticmethod
    def planner_from_base(
        reference: Trajectory = reference.small_circle,
        weights: JaxMpccPlannerWeights = JaxMpccPlannerWeights(),
        sampling: JaxSamplingOptions = JaxSamplingOptions(),
    ) -> JaxMpccPlannerConfiguration:
        # NOTE: Type Checkers like Pyright won't be able to infer complex types, so you may
        # need to help them with an explicit annotation.
        planner: Planner = mppi.jax.base(
            model=(
                augmented_model := AugmentedModel.of(
                    physical=model.jax.bicycle.dynamical(
                        time_step_size=(dt := 0.1),
                        wheelbase=(L := 2.5),
                        speed_limits=(0.0, 15.0),
                        steering_limits=(-0.5, 0.5),
                        acceleration_limits=(-3.0, 3.0),
                    ),
                    virtual=model.jax.integrator.dynamical(
                        time_step_size=dt,
                        state_limits=(0, reference.path_length),
                        velocity_limits=(0, 15),
                    ),
                    state=types.jax.augmented.state,
                    batch=types.jax.augmented.state_batch,
                )
            ),
            cost_function=costs.jax.combined(
                contouring_cost := costs.jax.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_extractor := extract.from_virtual(path_parameter)
                    ),
                    position_extractor=(
                        position_extractor := extract.from_physical(position)
                    ),
                    weight=weights.contouring,
                ),
                costs.jax.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=weights.lag,
                ),
                costs.jax.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(path_velocity),
                    time_step_size=dt,
                    weight=weights.progress,
                ),
                costs.jax.comfort.control_smoothing(weights=weights.control_smoothing),
            ),
            sampler=AugmentedSampler.of(
                physical=sampler.jax.gaussian(
                    standard_deviation=sampling.physical_standard_deviation,
                    rollout_count=sampling.rollout_count,
                    to_batch=types.jax.bicycle.control_input_batch.create,
                    key=sampling.physical_key,
                ),
                virtual=sampler.jax.gaussian(
                    standard_deviation=sampling.virtual_standard_deviation,
                    rollout_count=sampling.rollout_count,
                    to_batch=types.jax.simple.control_input_batch.create,
                    key=sampling.virtual_key,
                ),
                batch=types.jax.augmented.control_input_batch,
            ),
        )

        return JaxMpccPlannerConfiguration(
            reference=reference,
            planner=planner,
            model=augmented_model,
            contouring_cost=contouring_cost,
            wheelbase=L,
        )

    @staticmethod
    def planner_from_augmented(
        *,
        reference: Trajectory = reference.small_circle,
        obstacles: ObstacleStateProvider = obstacles.none,
        weights: JaxMpccPlannerWeights = JaxMpccPlannerWeights(),
        sampling: JaxSamplingOptions = JaxSamplingOptions(),
        use_covariance_propagation: bool = False,
    ) -> JaxMpccPlannerConfiguration:
        obstacles = obstacles.with_time_step(dt := 0.1).with_predictor(
            predictor.curvilinear(
                horizon=HORIZON,
                model=model.jax.bicycle.obstacle(
                    time_step_size=dt, wheelbase=(L := 2.5)
                ),
                prediction=bicycle_to_obstacle_states,
                propagator=propagator.jax.linear(
                    time_step_size=dt,
                    initial_covariance=propagator.jax.covariance.constant_variance(
                        position_variance=0.01, velocity_variance=1.0
                    ),
                    padding=propagator.padding(to_dimension=3, epsilon=1e-9),
                )
                if use_covariance_propagation
                else None,
            )
        )

        planner, augmented_model = mppi.jax.augmented(
            models=(
                model.jax.bicycle.dynamical(
                    time_step_size=dt,
                    wheelbase=L,
                    speed_limits=(0.0, 15.0),
                    steering_limits=(-0.5, 0.5),
                    acceleration_limits=(-3.0, 3.0),
                ),
                model.jax.integrator.dynamical(
                    time_step_size=dt,
                    state_limits=(0, reference.path_length),
                    velocity_limits=(0, 15),
                ),
            ),
            samplers=(
                sampler.jax.gaussian(
                    standard_deviation=sampling.physical_standard_deviation,
                    rollout_count=sampling.rollout_count,
                    to_batch=types.jax.bicycle.control_input_batch.create,
                    key=sampling.physical_key,
                ),
                sampler.jax.gaussian(
                    standard_deviation=sampling.virtual_standard_deviation,
                    rollout_count=sampling.rollout_count,
                    to_batch=types.jax.simple.control_input_batch.create,
                    key=sampling.virtual_key,
                ),
            ),
            cost=costs.jax.combined(
                contouring_cost := costs.jax.tracking.contouring(
                    reference=reference,
                    path_parameter_extractor=(
                        path_extractor := extract.from_virtual(path_parameter)
                    ),
                    position_extractor=(
                        position_extractor := extract.from_physical(position)
                    ),
                    weight=weights.contouring,
                ),
                costs.jax.tracking.lag(
                    reference=reference,
                    path_parameter_extractor=path_extractor,
                    position_extractor=position_extractor,
                    weight=weights.lag,
                ),
                costs.jax.tracking.progress(
                    path_velocity_extractor=extract.from_virtual(path_velocity),
                    time_step_size=dt,
                    weight=weights.progress,
                ),
                costs.jax.comfort.control_smoothing(weights=weights.control_smoothing),
                costs.jax.safety.collision(
                    obstacle_states=obstacles,
                    sampler=create_obstacles.sampler.jax.gaussian(),
                    distance=(
                        circles_distance := distance.jax.circles(
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
                    metric=risk.jax.mean_variance(gamma=0.5, sample_count=10)
                    if use_covariance_propagation
                    else None,
                ),
            ),
            state=types.jax.augmented.state,
            state_batch=types.jax.augmented.state_batch,
            input_batch=types.jax.augmented.control_input_batch,
        )

        return JaxMpccPlannerConfiguration(
            reference=reference,
            planner=planner,
            model=augmented_model,
            contouring_cost=contouring_cost,
            wheelbase=L,
            distance=circles_distance,
            obstacles=obstacles,
        )
