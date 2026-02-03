# trajax

[![Pipeline Status](https://gitlab.com/risk-metrics/trajax/badges/main/pipeline.svg)](https://gitlab.com/risk-metrics/trajax/-/pipelines) [![Coverage](https://codecov.io/gl/risk-metrics/trajax/graph/badge.svg)](https://codecov.io/gl/risk-metrics/trajax) [![Benchmark](https://bencher.dev/perf/trajax?testbed=gitlab-ci&key=true)](https://bencher.dev/perf/trajax) [![PyPI](https://img.shields.io/pypi/v/trajax)](https://pypi.org/project/trajax/) [![Python](https://img.shields.io/pypi/pyversions/trajax)](https://pypi.org/project/trajax/) [![License](https://img.shields.io/pypi/l/trajax)](https://gitlab.com/risk-metrics/trajax/-/blob/main/LICENSE)

A sampling-based trajectory planning library with NumPy and JAX backends for building MPPI (Model Predictive Path Integral) planners.

## Features

- **Dual Backend**: Identical APIs for NumPy (prototyping) and JAX (GPU acceleration)
- **MPPI Planning**: Sampling-based trajectory optimization with configurable cost functions
- **MPCC Support**: Model Predictive Contouring Control for path-following tasks
- **Modular Design**: Composable cost functions, samplers, and dynamical models
- **Risk-Aware Planning**: Integration with risk metrics (CVaR, VaR, entropic risk)
- **Obstacle Avoidance**: Circle and polygon collision checking with motion prediction

## Installation

```bash
pip install trajax
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add trajax
```

For GPU acceleration (Linux only):

```bash
pip install trajax[cuda]
```

## Quick Start

```python
from trajax.numpy import mppi, model, sampler, trajectory, types, extract
from numtypes import array

# Define position extractor for the cost function
def position(states):
    return types.positions(x=states.positions.x(), y=states.positions.y())

# Define the reference path to follow
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

# control.optimal - the optimal control sequence
# control.nominal - the updated nominal for the next iteration
```

## Switching to JAX

Replace imports to use GPU acceleration:

```python
# Change this:
from trajax.numpy import mppi, model, sampler, trajectory, types, extract

# To this:
from trajax.jax import mppi, model, sampler, trajectory, types, extract
```

All APIs remain identical between backends.

## Documentation

- **[Getting Started](https://risk-metrics.gitlab.io/trajax/guide/getting-started/)** — Installation and first planner
- **[Core Concepts](https://risk-metrics.gitlab.io/trajax/guide/concepts/)** — Understand MPPI and library architecture
- **[API Reference](https://risk-metrics.gitlab.io/trajax/api/)** — Complete API documentation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         MPPI Planner                        │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│   Sampler   │    Model    │    Cost     │     Filter       │
│  (Gaussian  │  (Bicycle,  │  (Tracking, │   (Savitzky-     │
│   Halton)   │  Integrator)│   Safety)   │    Golay)        │
└─────────────┴─────────────┴─────────────┴──────────────────┘
```

### Available Components

| Category | Components |
|----------|------------|
| **Models** | Kinematic bicycle, Unicycle, Integrator |
| **Samplers** | Gaussian, Halton-spline |
| **Costs** | Contouring, Lag, Progress, Collision, Boundary, Control smoothing |
| **Trajectories** | Waypoints (spline), Line |
| **Risk Metrics** | Expected value, Mean-variance, VaR, CVaR, Entropic risk |

## Requirements

- Python ≥ 3.13
- NumPy, JAX, SciPy

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding style, and testing guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.
