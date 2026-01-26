# Metrics and Evaluation

trajax provides metrics for evaluating trajectory planning performance.

## Setting Up Metrics

```python
from trajax import collectors, metrics, types

# Create collector wrappers
mppi_collector = collectors.states.decorating(
    planner,
    transformer=types.simple.state_sequence.of_states,
)

obstacle_collector = collectors.obstacle_states.decorating(
    observer,
    transformer=types.obstacle_states.of_states,
)

# Create registries
collector_registry = collectors.registry(mppi_collector, obstacle_collector)

registry = metrics.registry(
    metrics.collision(
        distance_threshold=0.5,
        distance=distance_extractor,
    ),
    collectors=collector_registry,
)
```

## Collision Metric

Detects when the ego vehicle collides with obstacles:

```python
collision_metric = metrics.collision(
    distance_threshold=0.5,
    distance=distance_extractor,
)

results = registry.get(collision_metric)
results.distances           # Shape: (T, V)
results.min_distances       # Shape: (V,)
results.collisions          # Shape: (T, V)
results.collision_detected  # bool
```

## Task Completion Metric

Determines when the vehicle reaches its goal and tracks progress along the reference trajectory:

```python
from trajax.numpy import trajectory, types

task_metric = metrics.task_completion(
    reference=trajectory.line(
        start=(1.0, -1.0),
        end=(15.0, 10.0),
        path_length=999.0,
    ),
    distance_threshold=5.0,
    time_step_size=0.5,
    position_extractor=lambda states: types.positions(
        x=states.array[:, 0],
        y=states.array[:, 1],
    ),
)

results = registry.get(task_metric)
results.completion       # List[bool] - completion status at each time step
results.completed        # bool - True if goal reached
results.completion_time  # float (inf if not completed)
results.stretch          # float - ratio of traveled distance to optimal
results.completed_part   # float - proportion of trajectory completed (see below)
```

### Completed Part

The `completed_part` property tracks how much of the reference trajectory has been completed:

- **0.0** → at start
- **0.5** → halfway along trajectory
- **1.0** → at end of trajectory
- **1.5** → halfway through second lap (for looped trajectories)

Progress is clamped to [0, 1] within each lap. Laps are counted when the vehicle's
longitudinal position wraps around (drops by more than X% of the path length, with X=50 by default),
indicating a transition from the end back to the start of a looped trajectory.

## Querying Metrics

```python
# By instance (type-safe)
results = registry.get(collision_metric)

# By name
results = registry.get(collision_metric.name)
```

## Live Recomputation

Metrics automatically recompute when new data arrives:

```python
for step in range(horizon):
    mppi_collector.step(
        temperature=50.0,
        nominal_input=nominal_input,
        initial_state=current_state,
    )
    obstacle_collector.observe(obstacle_states)

    current_results = registry.get(collision_metric)
    print(f"Step {step}: collision={current_results.collision_detected}")
```

## Complete Example

```python
from trajax import collectors, metrics, types
from trajax.numpy import mppi, trajectory

planner = mppi.base(...)
observer = obstacle_observer(...)

mppi_collector = collectors.states.decorating(
    planner,
    transformer=types.simple.state_sequence.of_states,
)
obstacle_collector = collectors.obstacle_states.decorating(
    observer,
    transformer=types.obstacle_states.of_states,
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

for step in range(100):
    mppi_collector.step(
        temperature=50.0,
        nominal_input=nominal,
        initial_state=state,
    )
    obstacle_collector.observe(current_obstacles)

    if registry.get(collision_metric).collision_detected:
        print("Collision!")
        break

    if registry.get(task_metric).completed:
        print(f"Goal reached in {registry.get(task_metric).completion_time}s!")
        break
```
