# API Reference

This section provides documentation for the trajax public API.

## Module Overview

| Module | Purpose | Key Factories |
|--------|---------|---------------|
| [`mppi`](mppi.md) | MPPI planners | `mppi.base()`, `mppi.augmented()`, `mppi.mpcc()` |
| [`model`](model.md) | Dynamics models | `model.bicycle.dynamical()`, `model.integrator()` |
| [`trajectory`](trajectory.md) | Reference paths | `trajectory.waypoints()`, `trajectory.line()` |
| [`boundary`](boundary.md) | Corridor constraints | `boundary.fixed_width()`, `boundary.piecewise_fixed_width()` |
| [`costs`](costs.md) | Cost functions | `costs.tracking.*`, `costs.safety.*`, `costs.comfort.*` |
| [`sampler`](sampler.md) | Control samplers | `sampler.gaussian()`, `sampler.halton()` |
| [`obstacles`](obstacles.md) | Obstacle handling | `obstacles.observer()`, `obstacles.predictor()` |
| [`types`](types.md) | Protocols and types | `types.positions()`, `types.path_parameters()` |
| [`metrics`](metrics.md) | Evaluation metrics | `metrics.collision()`, `metrics.task_completion()` |
| [`collectors`](collectors.md) | Data collection | `collectors.states.decorating()` |

## Backend Namespaces

All factory functions are accessed through backend namespaces:

```python
from trajax import mppi, model, sampler, costs, trajectory, boundary, types

# NumPy backend (prototyping)
planner = mppi.numpy.base(...)
bicycle = model.numpy.bicycle.dynamical(...)
reference = trajectory.numpy.waypoints(...)

# JAX backend (production)
planner = mppi.jax.base(...)
bicycle = model.jax.bicycle.dynamical(...)
reference = trajectory.jax.waypoints(...)
```

## Quick Reference

### Creating an MPPI Planner

```python
from trajax.numpy import mppi, model, sampler, costs, trajectory, types

# 1. Define reference path
reference = trajectory.waypoints(
    points=array([[0, 0], [10, 0], [20, 5], [30, 5]]),
    path_length=35.0,
)

# 2. Create dynamics model
bicycle = model.bicycle.dynamical(
    time_step_size=0.1,
    wheelbase=2.5,
    acceleration_limits=(-3.0, 3.0),
    steering_limits=(-0.5, 0.5),
)

# 3. Define cost function
cost = costs.combined(
    costs.tracking.contouring(reference=reference, weight=50.0, ...),
    costs.tracking.lag(reference=reference, weight=100.0, ...),
)

# 4. Create sampler
control_sampler = sampler.gaussian(seed=42)

# 5. Build planner
planner = mppi.base(
    model=bicycle,
    cost_function=cost,
    sampler=control_sampler,
    horizon=50,
    rollout_count=512,
)
```

### MPCC Shortcut

```python
# All-in-one MPCC planner
planner = mppi.mpcc(
    horizon=50,
    rollout_count=512,
    reference=reference,
    boundary=corridor,
    physical_model=bicycle,
    contouring_weight=50.0,
    lag_weight=100.0,
    progress_weight=1000.0,
)
```

## Conventions

### Signed Distances

Boundary distances follow this sign convention:

| Distance | Meaning |
|----------|---------|
| Positive | Inside valid region |
| Zero | On boundary |
| Negative | Outside (violation) |

### State Batch Shape

State batches have shape `(T, D_x, M)`:

- `T`: Time horizon (steps)
- `D_x`: State dimension
- `M`: Number of rollouts

### Lateral Direction

The "left" side is the positive lateral direction (90° counter-clockwise from path heading).

## Type Annotations

trajax uses generic type parameters for backend-agnostic code:

```python
from trajax import Mppi, CostFunction, Trajectory

def run_simulation[StateT, InputT](
    planner: Mppi[StateT, InputT],
    cost: CostFunction,
    reference: Trajectory,
) -> None:
    ...
```
