# Metrics and Evaluation

Evaluation metrics measure planning performance after (or during) a simulation. They operate on data collected by [decorating](../api/collectors.md) the planner and obstacle observer.

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

Reports contouring and lag error over the trajectory. Requires the `contouring_cost` and `lag_cost` objects returned by [`mppi.mpcc`](concepts.md#mpcc-model-predictive-contouring-control):

```python
planner, augmented_model, contouring_cost, lag_cost = mppi.mpcc(...)

error_metric = metrics.mpcc_error(contouring=contouring_cost, lag=lag_cost)
# ... register and run simulation ...
result = registry.get(error_metric)
result.contouring      # (T,) — contouring error at each time step
result.lag             # (T,) — lag error at each time step
result.max_contouring  # float — peak absolute contouring error
result.max_lag         # float — peak absolute lag error
```

### Constraint Violation

Reports boundary distances and flags violations where the vehicle leaves the corridor:

```python
violation_metric = metrics.constraint_violation(
    reference=reference,
    boundary=corridor,
    position_extractor=position_extractor,
)
# ... register and run simulation ...
result = registry.get(violation_metric)
result.lateral_deviations  # (T,) — lateral offset from the reference
result.boundary_distances  # (T,) — signed distance to corridor edge
result.violations          # (T,) — boolean flags where boundary_distance ≤ 0
result.violation_detected  # bool — True if any violation occurred
```

### Comfort

Reports lateral acceleration and jerk relative to the reference trajectory:

```python
comfort_metric = metrics.comfort(
    reference=reference,
    time_step_size=0.1,
    position_extractor=position_extractor,
)
# ... register and run simulation ...
result = registry.get(comfort_metric)
result.lateral_acceleration  # (T,) — lateral acceleration at each step
result.lateral_jerk          # (T,) — lateral jerk at each step
```

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
