# trajax

> **Primary repository:** [gitlab.com/risk-metrics/trajax](https://gitlab.com/risk-metrics/trajax) — the [GitHub mirror](https://github.com/zuka011/Trajax) exists for Colab notebook support.

[![Pipeline Status](https://gitlab.com/risk-metrics/trajax/badges/main/pipeline.svg)](https://gitlab.com/risk-metrics/trajax/-/pipelines) [![Coverage](https://codecov.io/gl/risk-metrics/trajax/graph/badge.svg?token=7O08BEVTAA)](https://codecov.io/gl/risk-metrics/trajax) [![Benchmarks](https://img.shields.io/badge/benchmarks-bencher.dev-blue)](https://bencher.dev/perf/trajax) [![PyPI](https://img.shields.io/pypi/v/trajax)](https://pypi.org/project/trajax/) [![Python](https://img.shields.io/pypi/pyversions/trajax)](https://pypi.org/project/trajax/) [![License](https://img.shields.io/pypi/l/trajax)](https://gitlab.com/risk-metrics/trajax/-/blob/main/LICENSE)

Sampling-based trajectory planning for autonomous systems. Provides composable building blocks — dynamics models, cost functions, samplers, and risk metrics — so you can assemble a complete MPPI planner in a few lines and iterate on the parts that matter for your problem.

## Installation

```bash
pip install trajax          # NumPy + JAX (CPU)
pip install trajax[cuda]    # JAX with GPU support (Linux)
```

Requires Python ≥ 3.13.

## Quick Start

MPPI planner with MPCC (Model Predictive Contouring Control) for path tracking, using a kinematic bicycle model:

```python
from trajax.numpy import mppi, model, sampler, trajectory, types, extract
from numtypes import array

def position(states):
    return types.positions(x=states.positions.x(), y=states.positions.y())

reference = trajectory.waypoints(
    points=array([[0, 0], [10, 0], [20, 5], [30, 5]], shape=(4, 2)),
    path_length=35.0,
)

planner, augmented_model, _, _ = mppi.mpcc(
    model=model.bicycle.dynamical(
        time_step_size=0.1, wheelbase=2.5,
        speed_limits=(0.0, 15.0), steering_limits=(-0.5, 0.5),
        acceleration_limits=(-3.0, 3.0),
    ),
    sampler=sampler.gaussian(
        standard_deviation=array([0.5, 0.2], shape=(2,)),
        rollout_count=256,
        to_batch=types.bicycle.control_input_batch.create, seed=42,
    ),
    reference=reference,
    position_extractor=extract.from_physical(position),
    config={
        "weights": {"contouring": 50.0, "lag": 100.0, "progress": 1000.0},
        "virtual": {"velocity_limits": (0.0, 15.0)},
    },
)

state = types.augmented.state.of(
    physical=types.bicycle.state.create(x=0.0, y=0.0, heading=0.0, speed=0.0),
    virtual=types.simple.state.zeroes(dimension=1),
)
nominal = types.augmented.control_input_sequence.of(
    physical=types.bicycle.control_input_sequence.zeroes(horizon=30),
    virtual=types.simple.control_input_sequence.zeroes(horizon=30, dimension=1),
)

for _ in range(200):
    control = planner.step(temperature=50.0, nominal_input=nominal, initial_state=state)
    state = augmented_model.step(inputs=control.optimal, state=state)
    nominal = control.nominal
```

<!-- TODO: Replace with simulation GIF -->

To use JAX (GPU), change `from trajax.numpy` to `from trajax.jax`. The API is identical.

## Features

See the [feature overview](https://risk-metrics.gitlab.io/trajax/guide/features/) for the full list of supported components, backend coverage, and roadmap.

## Documentation

| | |
|---|---|
| [Getting Started](https://risk-metrics.gitlab.io/trajax/guide/getting-started/) | Installation, first planner, simulation loop |
| [User Guide](https://risk-metrics.gitlab.io/trajax/guide/concepts/) | MPPI concepts, cost design, obstacles, boundaries, risk metrics |
| [Examples](https://risk-metrics.gitlab.io/trajax/guide/examples/) | Interactive visualizations of MPCC scenarios |
| [API Reference](https://risk-metrics.gitlab.io/trajax/api/) | Factory functions and protocol documentation |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
