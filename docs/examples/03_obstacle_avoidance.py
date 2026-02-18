"""MPCC path following with static obstacle avoidance.

Extends the previous examples by placing three static obstacles along the
path. Uses circle-based signed distance for collision cost and a curvilinear
predictor for obstacle motion forecasting.
"""

from dataclasses import dataclass

import numpy as np
from numtypes import array

from trajax import Circles, access, collectors, metrics
from trajax.numpy import (
    boundary,
    costs,
    distance,
    extract,
    filters,
    model,
    mppi,
    obstacles as create_obstacles,
    predictor,
    sampler,
    trajectory,
    types,
)
from trajax_visualizer import MpccSimulationResult

# ── Type aliases ──────────────────────────────────────────────────────────── #

type BicycleStateBatch = types.bicycle.StateBatch
type ObstacleStates = types.Obstacle2dPoses

# ── Constants ─────────────────────────────────────────────────────────────── #

HORIZON = 30
DT = 0.1
WHEELBASE = 2.5
VEHICLE_WIDTH = 1.2
TEMPERATURE = 50.0
ROLLOUT_COUNT = 512
MAX_STEPS = 350

# ── Extractors ────────────────────────────────────────────────────────────── #


def position(states: BicycleStateBatch) -> types.Positions:
    return types.positions(x=states.positions.x(), y=states.positions.y())


def heading(states: BicycleStateBatch) -> types.Headings:
    return types.headings(heading=states.heading())


class BicyclePredictionCreator:
    def __call__(
        self, *, states: types.bicycle.ObstacleStateSequences
    ) -> ObstacleStates:
        return states.pose()

    def empty(self, *, horizon: int) -> ObstacleStates:
        return types.obstacle_2d_poses.create(
            x=np.empty((horizon, 0)),
            y=np.empty((horizon, 0)),
            heading=np.empty((horizon, 0)),
        )


class ObstaclePositionExtractor:
    def of_states_for_time_step(self, states, /):
        return states.positions()

    def of_states(self, states, /):
        return states.positions()


# ── Reference & obstacles ─────────────────────────────────────────────────── #

# --8<-- [start:reference]
REFERENCE = trajectory.waypoints(
    points=array(
        [
            [0.0, 0.0],
            [15.0, 0.0],
            [30.0, 5.0],
            [45.0, 5.0],
            [60.0, 0.0],
        ],
        shape=(5, 2),
    ),
    path_length=70.0,
)
# --8<-- [end:reference]


# ── Result ────────────────────────────────────────────────────────────────── #


@dataclass(frozen=True)
class Result:
    """Outcome of a planning simulation with obstacles."""

    final_state: object
    visualization: MpccSimulationResult

    @property
    def progress(self) -> float:
        return float(self.final_state.virtual.array[0])

    @property
    def reached_goal(self) -> bool:
        return self.progress >= REFERENCE.path_length * 0.7


# ── Setup & run ───────────────────────────────────────────────────────────── #


# --8<-- [start:setup]
def create():
    position_extractor = extract.from_physical(position)

    obstacle_simulator = create_obstacles.static(
        positions=array(
            [[20.0, 2.5], [35.0, 7.5], [50.0, 2.5]],
            shape=(3, 2),
        ),
        headings=array([0.0, np.pi / 4, -np.pi / 6], shape=(3,)),
    )

    obstacles_provider = create_obstacles.provider.predicting(
        predictor=predictor.curvilinear(
            horizon=HORIZON,
            model=model.bicycle.obstacle(time_step_size=DT, wheelbase=WHEELBASE),
            estimator=model.bicycle.estimator.finite_difference(
                time_step_size=DT, wheelbase=WHEELBASE
            ),
            prediction=BicyclePredictionCreator(),
        ),
        history=types.obstacle_states_running_history.empty(
            creator=types.obstacle_2d_poses,
            horizon=2,
            obstacle_count=obstacle_simulator.obstacle_count,
        ),
        id_assignment=create_obstacles.id_assignment.hungarian(
            position_extractor=ObstaclePositionExtractor(),
            cutoff=10.0,
        ),
    )

    ego_circles = Circles(
        origins=array([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]], shape=(V := 3, 2)),
        radii=array([0.8, 0.8, 0.8], shape=(V,)),
    )

    circles_distance = distance.circles(
        ego=ego_circles,
        obstacle=ego_circles,
        position_extractor=position_extractor,
        heading_extractor=extract.from_physical(heading),
        obstacle_position_extractor=lambda states: states.positions(),
        obstacle_heading_extractor=lambda states: states.headings(),
    )

    corridor = boundary.fixed_width(
        reference=REFERENCE,
        position_extractor=position_extractor,
        left=3.0,
        right=3.0,
    )

    planner, augmented_model, contouring_cost, lag_cost = mppi.mpcc(
        model=model.bicycle.dynamical(
            time_step_size=DT,
            wheelbase=WHEELBASE,
            speed_limits=(0.0, 15.0),
            steering_limits=(-0.5, 0.5),
            acceleration_limits=(-3.0, 3.0),
        ),
        sampler=sampler.gaussian(
            standard_deviation=array([0.5, 0.2], shape=(2,)),
            rollout_count=ROLLOUT_COUNT,
            to_batch=types.bicycle.control_input_batch.create,
            seed=42,
        ),
        costs=(
            costs.comfort.control_smoothing(
                weights=array([5.0, 20.0, 5.0], shape=(3,)),
            ),
            costs.safety.collision(
                obstacle_states=(
                    forecasts_collector := collectors.obstacle_forecasts.decorating(
                        obstacles_provider
                    )
                ),
                sampler=create_obstacles.sampler.gaussian(seed=44),
                distance=circles_distance,
                distance_threshold=array([0.5, 0.5, 0.5], shape=(V,)),
                weight=1500.0,
            ),
            costs.safety.boundary(
                distance=corridor,
                distance_threshold=0.25,
                weight=1000.0,
            ),
        ),
        reference=REFERENCE,
        position_extractor=position_extractor,
        config={
            "weights": {"contouring": 50.0, "lag": 100.0, "progress": 1000.0},
            "virtual": {"velocity_limits": (1.0, 15.0)},
        },
        filter_function=filters.savgol(window_length=11, polynomial_order=3),
    )
    # --8<-- [end:setup]

    obstacle_collector = collectors.obstacle_states.decorating(
        obstacles_provider, transformer=types.obstacle_2d_poses.of_states
    )

    planner = (
        trajectories_collector := collectors.trajectories.decorating(
            state_collector := collectors.states.decorating(
                planner,
                transformer=types.augmented.state_sequence.of_states(
                    physical=types.bicycle.state_sequence.of_states,
                    virtual=types.simple.state_sequence.of_states,
                ),
            ),
            model=augmented_model,
        )
    )

    registry = metrics.registry(
        error_metric := metrics.mpcc_error(contouring=contouring_cost, lag=lag_cost),
        collision_metric := metrics.collision(
            distance_threshold=0.0, distance=circles_distance
        ),
        collectors=collectors.registry(
            state_collector,
            trajectories_collector,
            obstacle_collector,
            forecasts_collector,
        ),
    )

    return (
        planner,
        augmented_model,
        registry,
        error_metric,
        collision_metric,
        obstacle_simulator.with_time_step_size(DT),
        obstacle_collector,
        corridor,
    )


# --8<-- [start:loop]
def run(
    planner,
    augmented_model,
    registry,
    error_metric,
    collision_metric,
    obstacle_simulator,
    obstacle_observer,
    corridor,
) -> Result:
    current_state = types.augmented.state.of(
        physical=types.bicycle.state.create(x=0.0, y=0.0, heading=0.0, speed=0.0),
        virtual=types.simple.state.zeroes(dimension=1),
    )
    nominal = types.augmented.control_input_sequence.of(
        physical=types.bicycle.control_input_sequence.zeroes(horizon=HORIZON),
        virtual=types.simple.control_input_sequence.zeroes(
            horizon=HORIZON, dimension=1
        ),
    )

    for step in range(MAX_STEPS):
        control = planner.step(
            temperature=TEMPERATURE,
            nominal_input=nominal,
            initial_state=current_state,
        )
        nominal = control.nominal
        current_state = augmented_model.step(
            inputs=control.optimal, state=current_state
        )
        obstacle_observer.observe(obstacle_simulator.step())

        if current_state.virtual.array[0] >= REFERENCE.path_length * 0.7:
            break
    # --8<-- [end:loop]

    trajectories = registry.data(access.trajectories.require())
    errors = registry.get(error_metric)

    return Result(
        final_state=current_state,
        visualization=MpccSimulationResult(
            reference=REFERENCE,
            states=registry.data(access.states.require()),
            optimal_trajectories=[it.optimal for it in trajectories],
            nominal_trajectories=[it.nominal for it in trajectories],
            contouring_errors=errors.contouring,
            lag_errors=errors.lag,
            time_step_size=DT,
            wheelbase=WHEELBASE,
            vehicle_width=VEHICLE_WIDTH,
            max_contouring_error=5.0,
            max_lag_error=7.5,
            obstacles=registry.data(access.obstacle_states.require()),
            obstacle_forecasts=registry.data(access.obstacle_forecasts.require()),
            boundary=corridor,
        ),
    )


SEED = "doc-static-obstacles"
MAX_CONTOURING_ERROR = 5.0
MAX_LAG_ERROR = 7.5
HAS_COLLISION_METRIC = True
GOAL_FRACTION = 0.7


if __name__ == "__main__":
    components = create()
    result = run(*components)
    print(f"Path progress: {result.progress:.1f} / {REFERENCE.path_length}")
    print(f"Reached goal: {result.reached_goal}")
