# Getting Started

This guide walks you through building an MPPI planner that follows a reference path.

## Installation

```bash
pip install trajax
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add trajax
```

## Prerequisites

- Python 3.13 or later

## The Complete Example

```python
from trajax.numpy import (
    mppi, model, sampler, costs, trajectory, types, extract, filters
)
from trajax.states import AugmentedModel, AugmentedSampler
from numtypes import array

# Define extractors
def path_parameter(states):
    return types.path_parameters(states.array[:, 0, :])

def path_velocity(inputs):
    return inputs.array[:, 0, :]

def position(states):
    return types.positions(x=states.positions.x(), y=states.positions.y())

# Create reference trajectory
reference = trajectory.waypoints(
    points=array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 5.0],
        [25.0, 15.0],
        [20.0, 25.0],
    ], shape=(5, 2)),
    path_length=50.0,
)

# Create kinematic bicycle model
bicycle = model.bicycle.dynamical(
    time_step_size=0.1,
    wheelbase=2.5,
    speed_limits=(0.0, 15.0),
    steering_limits=(-0.5, 0.5),
    acceleration_limits=(-3.0, 3.0),
)

# Create integrator for the virtual path parameter
virtual_model = model.integrator.dynamical(
    time_step_size=0.1,
    state_limits=(0, reference.path_length),
    velocity_limits=(1.0, 15.0),
)

# Combine into augmented model
augmented_model = AugmentedModel.of(
    physical=bicycle,
    virtual=virtual_model,
    state=types.augmented.state,
    sequence=types.augmented.state_sequence,
    batch=types.augmented.state_batch,
)

# Create cost function
cost = costs.combined(
    costs.tracking.contouring(
        reference=reference,
        path_parameter_extractor=extract.from_virtual(path_parameter),
        position_extractor=extract.from_physical(position),
        weight=50.0,
    ),
    costs.tracking.lag(
        reference=reference,
        path_parameter_extractor=extract.from_virtual(path_parameter),
        position_extractor=extract.from_physical(position),
        weight=100.0,
    ),
    costs.tracking.progress(
        path_velocity_extractor=extract.from_virtual(path_velocity),
        time_step_size=0.1,
        weight=1000.0,
    ),
)

# Create combined sampler
combined_sampler = AugmentedSampler.of(
    physical=sampler.gaussian(
        standard_deviation=array([0.5, 0.2], shape=(2,)),
        rollout_count=256,
        to_batch=types.bicycle.control_input_batch.create,
        seed=42,
    ),
    virtual=sampler.gaussian(
        standard_deviation=array([2.0], shape=(1,)),
        rollout_count=256,
        to_batch=types.simple.control_input_batch.create,
        seed=43,
    ),
    batch=types.augmented.control_input_batch,
)

# Create the MPPI planner
planner = mppi.base(
    model=augmented_model,
    cost_function=cost,
    sampler=combined_sampler,
    filter_function=filters.savgol(window_length=11, polynomial_order=3),
)

# Initialize state and controls
initial_state = types.augmented.state.of(
    physical=types.bicycle.state.create(x=0.0, y=0.0, heading=0.0, speed=0.0),
    virtual=types.simple.state.zeroes(dimension=1),
)
nominal_input = types.augmented.control_input_sequence.of(
    physical=types.bicycle.control_input_sequence.zeroes(horizon=30),
    virtual=types.simple.control_input_sequence.zeroes(horizon=30, dimension=1),
)

# Run the planning loop
current_state = initial_state
for step in range(150):
    control = planner.step(
        temperature=50.0,
        nominal_input=nominal_input,
        initial_state=current_state,
    )
    
    nominal_input = control.nominal
    current_state = augmented_model.step(
        inputs=control.optimal, state=current_state
    )
    
    progress = current_state.virtual.array[0]
    if progress >= reference.path_length * 0.9:
        print(f"Reached goal in {step + 1} steps!")
        break
```

## Augmented State

MPCC tracks a "virtual" path parameter $\phi$ alongside the vehicle state:

- **Physical state**: Vehicle position, heading, speed $[x, y, \theta, v]$
- **Virtual state**: Path parameter $[\phi]$
- **Physical controls**: Acceleration, steering $[a, \delta]$
- **Virtual control**: Path velocity $[\dot{\phi}]$

## Cost Function Components

| Cost | Purpose |
|------|---------|
| **Contouring** | Penalizes lateral deviation from path |
| **Lag** | Penalizes falling behind the path parameter |
| **Progress** | Rewards forward motion along path |

## Shortcut: `mppi.mpcc()`

For MPCC path-following, there's a simpler factory:

```python
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

## Next Steps

- [Core Concepts](concepts.md) — Understand the MPPI algorithm
- [MPPI Planning](mppi.md) — Configure temperature and filtering
- [Obstacle Handling](obstacles.md) — Add collision avoidance
- [Cost Function Design](costs.md) — Create custom objectives
