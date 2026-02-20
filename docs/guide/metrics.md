# Metrics and Evaluation

Evaluation metrics measure planning performance after (or during) a simulation. They operate on data collected by decorating the planner and obstacle observer.

## Setup

Wrap the planner and obstacle observer with collectors, then register the metrics you want:

```python
from faran import collectors, metrics

mppi_collector = collectors.states.decorating(
    planner,
    transformer=types.simple.state_sequence.of_states,
)

obstacle_collector = collectors.obstacle_states.decorating(
    observer,
    transformer=types.obstacle_2d_poses.of_states,
)

registry = metrics.registry(
    collision_metric := metrics.collision(
        distance_threshold=0.5,
        distance=distance_extractor,
    ),
    task_metric := metrics.task_completion(
        reference=reference,
        distance_threshold=2.0,
        time_step_size=0.1,
        position_extractor=lambda s: types.positions(x=s.array[:, 0], y=s.array[:, 1]),
    ),
    collectors=collectors.registry(mppi_collector, obstacle_collector),
)
```

## Available Metrics

### Collision

Detects collisions and reports minimum distances:

```python
result = registry.get(collision_metric)
result.distances           # (T, V) — distance per time step per vehicle part
result.min_distances       # (V,) — minimum over time
result.collisions          # (T, V) — boolean collision flags
result.collision_detected  # bool — any collision at any time
```

### Task Completion

Tracks progress along the reference trajectory and goal arrival:

```python
result = registry.get(task_metric)
result.completed        # bool — True if goal reached
result.completion_time  # float — seconds to reach goal (inf if not reached)
result.stretch          # float — ratio of traveled distance to optimal
result.completed_part   # float — fraction of reference completed (0 to 1+)
```

`completed_part` can exceed 1.0 for looped trajectories, counting additional laps.

### MPCC Error

Reports contouring and lag error over the trajectory. Requires the `contouring_cost` and `lag_cost` objects returned by `mppi.mpcc`.

### Constraint Violation

Reports boundary and input limit violations.

### Comfort

Reports jerk, lateral acceleration, and smoothness metrics.

## Live Evaluation

Metrics recompute automatically when new data arrives. You can query them inside the planning loop:

```python
for step in range(horizon):
    mppi_collector.step(
        temperature=50.0, nominal_input=nominal, initial_state=state,
    )
    obstacle_collector.observe(obstacle_states)

    if registry.get(collision_metric).collision_detected:
        print("Collision!")
        break
```
