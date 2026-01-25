# MPPI Planning

This guide covers how to configure MPPI planners.

## Factory Functions

trajax provides three factory functions:

| Factory | Use When |
|---------|----------|
| `mppi.base` | Custom MPC formulation |
| `mppi.augmented` | Augmented states (physical + virtual) |
| `mppi.mpcc` | Path-following with MPCC |

## `mppi.base`

The lowest-level factory. You provide all components:

```python
from trajax.numpy import mppi, model, sampler, costs, types
from numtypes import array

planner = mppi.base(
    model=model.bicycle.dynamical(
        time_step_size=0.1,
        wheelbase=2.5,
        speed_limits=(0.0, 15.0),
        steering_limits=(-0.5, 0.5),
        acceleration_limits=(-3.0, 3.0),
    ),
    cost_function=cost,
    sampler=sampler.gaussian(
        standard_deviation=array([0.5, 0.2], shape=(2,)),
        rollout_count=256,
        to_batch=types.bicycle.control_input_batch.create,
        seed=42,
    ),
)
```

### Optional Configuration

```python
from trajax.numpy import filters, update, padding

planner = mppi.base(
    model=bicycle,
    cost_function=cost,
    sampler=control_sampler,
    planning_interval=1,
    filter_function=filters.savgol(window_length=11, polynomial_order=3),
    update_function=update.use_optimal_control(),
    padding_function=padding.zero(),
)
```

## `mppi.augmented`

Combines multiple models (e.g., physical + virtual states):

```python
from trajax.numpy import mppi, model, sampler, costs, types
from trajax.states import extract
from numtypes import array

planner, augmented_model = mppi.augmented(
    models=(
        model.bicycle.dynamical(...),
        model.integrator.dynamical(
            time_step_size=0.1,
            state_limits=(0, 100.0),
            velocity_limits=(1.0, 15.0),
        ),
    ),
    samplers=(physical_sampler, virtual_sampler),
    cost=cost,
    state=types.augmented.state,
    state_sequence=types.augmented.state_sequence,
    state_batch=types.augmented.state_batch,
    input_batch=types.augmented.control_input_batch,
)
```

## `mppi.mpcc`

Highest-level factory for MPCC path following:

```python
from trajax.numpy import mppi, model, sampler, trajectory, types, extract
from numtypes import array

def position(states):
    return types.positions(x=states.positions.x(), y=states.positions.y())

reference = trajectory.waypoints(
    points=array([[0, 0], [10, 0], [20, 5], [30, 5]], shape=(4, 2)),
    path_length=35.0,
)

planner, augmented_model, contouring_cost, lag_cost = mppi.mpcc(
    model=model.bicycle.dynamical(...),
    sampler=sampler.gaussian(...),
    reference=reference,
    position_extractor=extract.from_physical(position),
    config={
        "weights": {"contouring": 50.0, "lag": 100.0, "progress": 1000.0},
        "virtual": {"velocity_limits": (0.0, 15.0)},
    },
)
```

## Running the Planner

```python
control = planner.step(
    temperature=50.0,
    nominal_input=nominal_input,
    initial_state=current_state,
)

# control.optimal: The weighted-average control sequence
# control.nominal: Updated warm-start for next iteration
```

### Planning Loop

```python
current_state = initial_state
nominal_input = initial_nominal

for step in range(max_steps):
    control = planner.step(
        temperature=50.0,
        nominal_input=nominal_input,
        initial_state=current_state,
    )
    
    current_state = model.step(inputs=control.optimal, state=current_state)
    nominal_input = control.nominal
```

## Temperature

The temperature parameter $\lambda$ controls exploration:

| Temperature | Behavior |
|-------------|----------|
| Low | Favor low-cost samples |
| High | Consider all samples more equally |

## Filtering

Smooths the optimal control sequence:

```python
from trajax.numpy import filters

planner = mppi.base(
    ...,
    filter_function=filters.savgol(window_length=11, polynomial_order=3),
)
```

## Sampler Seeding

Samplers are deterministic given a seed. Use different seeds for physical and virtual samplers:

```python
physical_sampler = sampler.gaussian(seed=42, ...)
virtual_sampler = sampler.gaussian(seed=43, ...)
```
