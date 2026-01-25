# Backends

trajax provides identical APIs for NumPy and JAX backends.

## Choosing a Backend

| Backend | Best For |
|---------|----------|
| NumPy | Prototyping, debugging |
| JAX | GPU acceleration, production |

## Importing

```python
# NumPy backend
from trajax.numpy import mppi, model, sampler, costs, trajectory, boundary, types

# JAX backend
from trajax.jax import mppi, model, sampler, costs, trajectory, boundary, types
```

Both expose identical factory functions. Pick one and use it consistently throughout your project.

## Consistency Requirement

All components in a pipeline must use the same backend:

```python
# ✅ Correct: All NumPy
reference = trajectory.numpy.waypoints(...)
corridor = boundary.numpy.fixed_width(reference=reference, ...)
cost = costs.numpy.tracking.contouring(reference=reference, ...)

# ❌ Wrong: Mixed
reference = trajectory.numpy.waypoints(...)
corridor = boundary.jax.fixed_width(reference=reference, ...)
```

## Converting Outputs

JAX arrays convert to NumPy with `np.asarray()`:

```python
import numpy as np

jax_control = jax_planner.step(...)
numpy_array = np.asarray(jax_control.optimal)
```

## Switching Backends

To switch a codebase from NumPy to JAX:

1. Change imports from `trajax.numpy` to `trajax.jax`
2. Ensure all components use the same backend
3. Add a warm-up call before timing-critical sections
4. Convert outputs to NumPy if needed downstream

The identical APIs make this a straightforward find-and-replace operation.
```

## Checking Backend Availability

```python
try:
    import jax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Choose backend based on availability
if JAX_AVAILABLE:
    from trajax import trajectory
    reference = trajectory.jax.line(...)
else:
    from trajax import trajectory
    reference = trajectory.numpy.line(...)
```
