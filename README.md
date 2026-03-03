<p align="center">
    <a href="https://risk-metrics.gitlab.io/faran/">
        <img src="./assets/logo.svg" width="250" alt="Faran Logo">
    </a>
</p>

# Faran: A Trajectory Planning Library for Autonomous Systems

> 
> The [GitHub mirror](https://github.com/zuka011/faran) of Faran exists to make the project easier to discover.  
> The main repository for issues and contributions can be found at [gitlab.com/risk-metrics/faran](https://gitlab.com/risk-metrics/faran)
> 

[![Pipeline Status](https://gitlab.com/risk-metrics/faran/badges/main/pipeline.svg)](https://gitlab.com/risk-metrics/faran/-/pipelines) [![Coverage](https://codecov.io/gl/risk-metrics/faran/graph/badge.svg?token=7O08BEVTAA)](https://codecov.io/gl/risk-metrics/faran) [![Benchmarks](https://img.shields.io/badge/benchmarks-bencher.dev-blue)](https://bencher.dev/perf/faran) [![PyPI](https://img.shields.io/pypi/v/faran)](https://pypi.org/project/faran/) [![Python](https://img.shields.io/pypi/pyversions/faran)](https://pypi.org/project/faran/) [![License](https://img.shields.io/pypi/l/faran)](https://gitlab.com/risk-metrics/faran/-/blob/main/LICENSE)

This library provides composable building blocks for creating trajectory planners for autonomous systems. Currently, Faran provides implementations of the Model-Predictive Path Integral (MPPI) algorithm for NumPy and JAX. Gradient-based methods, such as iLQR are not yet available, but in the roadmap. An additional package - `faran-visualizer` - provides a CLI for generating interactive HTML visualizations of planner behavior.

## Installation

```bash
pip install faran          # NumPy + JAX (CPU only)
pip install faran[cuda]    # JAX with GPU support
```

The visualizer CLI can be installed separately:

```bash
pip install faran-visualizer
```

## Quick Start

MPPI planner for the Model Predictive Contouring Control (MPCC) formulation, assuming a kinematic bicycle model for the system dynamics:

```python
from faran import access, collectors, metrics
from faran.numpy import mppi, model, sampler, trajectory, types, extract

import numpy as np

reference = trajectory.waypoints(
    points=np.array([[0, 0], [10, 0], [20, 5], [30, 0], [40, -5], [50, 0]]),
    path_length=35.0,
)

planner, augmented_model, contouring_cost, lag_cost = mppi.mpcc(
    model=model.bicycle.dynamical(
        time_step_size=0.1, wheelbase=2.5,
        speed_limits=(0.0, 15.0), steering_limits=(-0.5, 0.5),
        acceleration_limits=(-3.0, 3.0),
    ),
    sampler=sampler.gaussian(
        standard_deviation=np.array([0.5, 0.05]),
        rollout_count=256,
        to_batch=types.bicycle.control_input_batch.create, seed=42,
    ),
    reference=reference,
    position_extractor=extract.from_physical(lambda states: states.positions),
    config={
        "weights": {"contouring": 100.0, "lag": 100.0, "progress": 1000.0},
        "virtual": {"velocity_limits": (0.0, 15.0)},
    },
)
```

To see how the planner works, we can collect runtime data as follows:

```python
planner = collectors.states.decorating(
    planner,
    transformer=types.augmented.state_sequence.of_states(
        physical=types.bicycle.state_sequence.of_states,
        virtual=types.simple.state_sequence.of_states,
    ),
)
registry = metrics.registry(
    error_metric := metrics.mpcc_error(contouring=contouring_cost, lag=lag_cost),
    collectors=collectors.registry(planner),
)
```

Now we set up a dummy simulation loop.

```python
state = types.augmented.state.of(
    physical=types.bicycle.state.create(x=0.0, y=0.0, heading=0.0, speed=0.0),
    virtual=types.simple.state.zeroes(dimension=1),
)
nominal = types.augmented.control_input_sequence.of(
    physical=types.bicycle.control_input_sequence.zeroes(horizon=30),
    virtual=types.simple.control_input_sequence.zeroes(horizon=30, dimension=1),
)

for _ in range(100):
    control = planner.step(temperature=50.0, nominal_input=nominal, initial_state=state)
    state = augmented_model.step(inputs=control.optimal, state=state)
    nominal = control.nominal
```

Finally, we can visualize the results:

```python
import asyncio
from faran_visualizer import MpccSimulationResult, configure, visualizer

errors = registry.get(error_metric)
result = MpccSimulationResult(
    reference=reference,
    states=registry.data(access.states.require()),
    contouring_errors=errors.contouring,
    lag_errors=errors.lag,
    time_step_size=0.1,
    wheelbase=2.5,
)

configure(output_directory=".")
asyncio.run(visualizer.mpcc()(result, key="quickstart"))
```

![Quickstart visualization](./assets/quickstart.gif)

Switching `from faran.numpy` with `from faran.jax` will use the JAX backend instead. The JAX API is compatible with the NumPy version.

## Features

See the [feature overview](https://risk-metrics.gitlab.io/faran/guide/features/) for the full list of supported components, backend coverage, and roadmap.

## Documentation

|                                                                                |                                                                     |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------|
| [Getting Started](https://risk-metrics.gitlab.io/faran/guide/getting-started/) | Installation, first planner, simulation loop                        |
| [User Guide](https://risk-metrics.gitlab.io/faran/guide/concepts/)             | Planning concepts, cost design, obstacles, boundaries, risk metrics |
| [Examples](https://risk-metrics.gitlab.io/faran/guide/examples/)               | Interactive visualizations of MPCC scenarios                        |
| [API Reference](https://risk-metrics.gitlab.io/faran/api/)                     | Function signatures and technical documentation                     |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and [DESIGN.md](DESIGN.md).

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

MIT — see [LICENSE](LICENSE).
