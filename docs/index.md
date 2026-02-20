---
hide:
  - navigation
  - toc
---

# faran

Sampling-based trajectory planning for autonomous systems. The library provides composable building blocks — dynamics models, cost functions, samplers, and risk metrics — that you wire together into an MPPI planner. NumPy and JAX backends expose the same API; switch between them by changing one import line.

## Quick Start

An MPCC (Model Predictive Contouring Control) formulation tracking a reference path with a kinematic bicycle model:

<div class="grid" markdown>

```python
from faran.numpy import mppi, model, sampler, trajectory, types, extract
from numtypes import array

def position(states):
    return types.positions(
        x=states.positions.x(), y=states.positions.y()
    )

reference = trajectory.waypoints(
    points=array(
        [[0, 0], [10, 0], [20, 5], [30, 5]], shape=(4, 2)
    ),
    path_length=35.0,
)

planner, augmented_model, _, _ = mppi.mpcc(
    model=model.bicycle.dynamical(
        time_step_size=0.1, wheelbase=2.5,
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
        "weights": {
            "contouring": 50.0,
            "lag": 100.0,
            "progress": 1000.0,
        },
        "virtual": {"velocity_limits": (0.0, 15.0)},
    },
)

state = types.augmented.state.of(
    physical=types.bicycle.state.create(
        x=0.0, y=0.0, heading=0.0, speed=0.0
    ),
    virtual=types.simple.state.zeroes(dimension=1),
)
nominal = types.augmented.control_input_sequence.of(
    physical=types.bicycle.control_input_sequence.zeroes(
        horizon=30
    ),
    virtual=types.simple.control_input_sequence.zeroes(
        horizon=30, dimension=1
    ),
)

for _ in range(200):
    control = planner.step(
        temperature=50.0,
        nominal_input=nominal,
        initial_state=state,
    )
    state = augmented_model.step(
        inputs=control.optimal, state=state
    )
    nominal = control.nominal
```

<!-- TODO: Replace with simulation GIF -->
![MPCC simulation placeholder](https://via.placeholder.com/480x400?text=MPCC+Simulation)

</div>

To use JAX (GPU), change `from faran.numpy` to `from faran.jax`. Everything else stays the same.

## Installation

```bash
pip install faran          # NumPy + JAX (CPU)
pip install faran[cuda]    # JAX with GPU support (Linux)
```

Requires Python ≥ 3.13.

## Features

See the [feature overview](guide/features.md) for the full list of supported components, backend coverage, and roadmap.

## Documentation

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting Started**

    ---

    Install faran and run your first MPPI planner

    [:octicons-arrow-right-24: Getting started](guide/getting-started.md)

-   :material-book-open: **User Guide**

    ---

    MPPI concepts, cost design, obstacles, boundaries, risk metrics

    [:octicons-arrow-right-24: User guide](guide/concepts.md)

-   :material-play-box: **Examples**

    ---

    Interactive visualizations of MPCC scenarios

    [:octicons-arrow-right-24: Examples](guide/examples.md)

-   :material-code-tags: **API Reference**

    ---

    Factory functions and protocol documentation

    [:octicons-arrow-right-24: Reference](api/index.md)

</div>

## License

MIT
