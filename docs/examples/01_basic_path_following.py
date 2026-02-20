"""Basic MPCC path following with a kinematic bicycle model.

The planner tracks an S-curve reference trajectory using contouring, lag,
and progress costs with a Savitzky-Golay filter for control smoothing.
"""

from dataclasses import dataclass

from numtypes import array
from tqdm.auto import tqdm

from faran import MpccErrorMetricResult, access, collectors, metrics
from faran.numpy import (
    costs,
    extract,
    filters,
    model,
    mppi,
    sampler,
    trajectory,
    types,
)
from faran_visualizer import MpccSimulationResult

# ── Type aliases ──────────────────────────────────────────────────────────── #

type BicycleState = types.bicycle.State
type BicycleStateBatch = types.bicycle.StateBatch
type AugmentedState = types.augmented.State[BicycleState, types.simple.State]
type AugmentedInputSequence = types.augmented.ControlInputSequence[
    types.bicycle.ControlInputSequence, types.simple.ControlInputSequence
]

# ── Constants ─────────────────────────────────────────────────────────────── #

HORIZON = 30
DT = 0.1
WHEELBASE = 2.5
VEHICLE_WIDTH = 1.2
TEMPERATURE = 50.0
ROLLOUT_COUNT = 256
STEP_LIMIT = 150

# ── Extractors ────────────────────────────────────────────────────────────── #


def position(states: BicycleStateBatch) -> types.Positions:
    return types.positions(x=states.positions.x(), y=states.positions.y())


def heading(states: BicycleStateBatch) -> types.Headings:
    return types.headings(heading=states.heading())


# --8<-- [start:reference]
REFERENCE = trajectory.waypoints(
    points=array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 5.0],
            [25.0, 15.0],
            [20.0, 25.0],
        ],
        shape=(5, 2),
    ),
    path_length=50.0,
)
# --8<-- [end:reference]


# ── Result ────────────────────────────────────────────────────────────────── #


@dataclass(frozen=True)
class Result:
    """Outcome of a planning simulation."""

    final_state: AugmentedState
    visualization: MpccSimulationResult
    tracking_errors: MpccErrorMetricResult
    collision_detected: bool

    @property
    def progress(self) -> float:
        return float(self.final_state.virtual.array[0])

    @property
    def reached_goal(self) -> bool:
        return self.progress >= REFERENCE.path_length * 0.9


# ── Setup & run ───────────────────────────────────────────────────────────── #


# --8<-- [start:setup]
def create():
    position_extractor = extract.from_physical(position)

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
        ),
        reference=REFERENCE,
        position_extractor=position_extractor,
        config={
            "weights": {"contouring": 50.0, "lag": 100.0, "progress": 1000.0},
            "virtual": {"velocity_limits": (0.0, 15.0)},
        },
        filter_function=filters.savgol(window_length=11, polynomial_order=3),
    )
    # --8<-- [end:setup]

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
        collectors=collectors.registry(state_collector, trajectories_collector),
    )

    return planner, augmented_model, registry, error_metric


# --8<-- [start:loop]
def run(planner, augmented_model, registry, error_metric) -> Result:
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

    bar = tqdm(range(STEP_LIMIT), desc="Simulation", unit="step")
    for step in bar:
        control = planner.step(
            temperature=TEMPERATURE,
            nominal_input=nominal,
            initial_state=current_state,
        )
        nominal = control.nominal
        current_state = augmented_model.step(
            inputs=control.optimal, state=current_state
        )

        if current_state.virtual.array[0] >= REFERENCE.path_length * 0.9:
            bar.write(f"Reached goal at step {step + 1}.")
            break

        bar.set_postfix(progress=f"{current_state.virtual.array[0]:.1}%")
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
            max_contouring_error=2.5,
            max_lag_error=5.0,
        ),
        tracking_errors=errors,
        collision_detected=False,
    )


SEED = "doc-basic-path-following"
MAX_CONTOURING_ERROR = 2.5
MAX_LAG_ERROR = 5.0


# ── Visualization ──────────────────────────────────────────────────────────────────────────── #


# --8<-- [start:visualize]
async def visualize(result: Result) -> None:
    from faran_visualizer import configure, visualizer

    configure(output_directory=".")
    await visualizer.mpcc()(result.visualization, key="visualization")


# --8<-- [end:visualize]


if __name__ == "__main__":
    import asyncio

    planner, augmented_model, registry, error_metric = create()
    result = run(planner, augmented_model, registry, error_metric)
    print(f"Path progress: {result.progress:.1f} / {REFERENCE.path_length}")
    print(f"Reached goal: {result.reached_goal}")
    print(f"Collision detected: {result.collision_detected}")
    asyncio.run(visualize(result))
