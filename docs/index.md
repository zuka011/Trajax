# trajax

A sampling-based trajectory planning library with NumPy and JAX backends for building MPPI (Model Predictive Path Integral) planners.

## What This Library Does

trajax helps you build MPPI planners for autonomous systems. You define:

1. **A dynamical model** — how your system evolves given control inputs
2. **A cost function** — what behaviors to encourage or penalize  
3. **A sampler** — how to explore the control space

The planner samples candidate control sequences, simulates each one, scores them by cost, and returns a weighted combination of the best samples.

## Quick Start

```python
from trajax.numpy import mppi, model, sampler, trajectory, types, extract
from numtypes import array

# Define position extractor
def position(states):
    return types.positions(x=states.positions.x(), y=states.positions.y())

# Define the path to follow
reference = trajectory.waypoints(
    points=array([[0, 0], [10, 0], [20, 5], [30, 5]], shape=(4, 2)),
    path_length=35.0,
)

# Create an MPCC planner (path-following with contouring/lag costs)
planner, augmented_model, contouring_cost, lag_cost = mppi.mpcc(
    model=model.bicycle.dynamical(
        time_step_size=0.1,
        wheelbase=2.5,
        speed_limits=(0.0, 15.0),
        steering_limits=(-0.5, 0.5),
        acceleration_limits=(-3.0, 3.0),
    ),
    sampler=sampler.gaussian(
        standard_deviation=array([0.5, 0.2], shape=(2,)),
        rollout_count=256,
        to_batch=types.bicycle.control_input_batch.create,
        seed=42,
    ),
    reference=reference,
    position_extractor=extract.from_physical(position),
    config={
        "weights": {"contouring": 50.0, "lag": 100.0, "progress": 1000.0},
        "virtual": {"velocity_limits": (0.0, 15.0)},
    },
)

# Initialize state
initial_state = types.augmented.state.of(
    physical=types.bicycle.state.create(x=0.0, y=0.0, heading=0.0, speed=0.0),
    virtual=types.simple.state.zeroes(dimension=1),
)
nominal_input = types.augmented.control_input_sequence.of(
    physical=types.bicycle.control_input_sequence.zeroes(horizon=30),
    virtual=types.simple.control_input_sequence.zeroes(horizon=30, dimension=1),
)

# Run the planner
control = planner.step(
    temperature=50.0,
    nominal_input=nominal_input,
    initial_state=initial_state,
)
```

!!! tip "Lower-Level APIs Available"
    The `mppi.mpcc` factory is convenient for path-following, but you can use `mppi.base` or `mppi.augmented` to build planners with custom cost functions for other problem formulations.

## Installation

```bash
pip install trajax
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add trajax
```

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**
    
    ---
    
    Install trajax and create your first MPPI planner
    
    [:octicons-arrow-right-24: Quick start](guide/getting-started.md)

-   :material-book-open: **Core Concepts**
    
    ---
    
    Understand MPPI and the library architecture
    
    [:octicons-arrow-right-24: Learn more](guide/concepts.md)

-   :material-code-tags: **API Reference**
    
    ---
    
    Complete API documentation
    
    [:octicons-arrow-right-24: Reference](api/index.md)

-   :material-chart-line: **Visualizer**
    
    ---
    
    Interactive visualization for simulation results
    
    [:octicons-arrow-right-24: Visualize](guide/visualizer.md)

</div>

## License

MIT License
