# API Reference

## Module Overview

| Module | Purpose |
|--------|---------|
| [`mppi`](mppi.md) | MPPI planner factories |
| [`model`](model.md) | Dynamical models (bicycle, unicycle, integrator) |
| [`costs`](costs.md) | Cost functions (tracking, safety, comfort) |
| [`sampler`](sampler.md) | Control input samplers |
| [`trajectory`](trajectory.md) | Reference path definitions |
| [`boundary`](boundary.md) | Drivable corridor constraints |
| [`obstacles`](obstacles.md) | Obstacle state handling and sampling |
| [`predictor`](predictor.md) | Motion prediction and covariance propagation |
| [`collectors`](collectors.md) | Simulation data collection |
| [`metrics`](metrics.md) | Evaluation metrics |
| [`types`](types.md) | Protocols and type definitions |
| [`visualizer`](visualizer.md) | Interactive HTML visualizations |

## Backend Namespaces

All factory functions are accessed through backend namespaces:

```python
# NumPy backend
from trajax.numpy import mppi, model, sampler, costs, trajectory, boundary, types

# JAX backend
from trajax.jax import mppi, model, sampler, costs, trajectory, boundary, types
```

Both namespaces expose identical APIs. See [Backends](../guide/backends.md) for details.

## Conventions

### Signed Distances

| Distance | Meaning |
|----------|---------|
| Positive | Inside valid region |
| Zero | On boundary |
| Negative | Violation |

### State Batch Shape

State batches have shape $(T, D_x, M)$ where $T$ is the time horizon, $D_x$ the state dimension, and $M$ the number of rollouts. See [Notation](../guide/conventions.md) for the complete reference.
