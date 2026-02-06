# Backends

trajax exposes identical APIs under two backends. Switch by changing the import path.

```python
# NumPy — prototyping, debugging, no GPU dependency
from trajax.numpy import mppi, model, sampler, costs, trajectory, boundary, types

# JAX — GPU acceleration, JIT compilation
from trajax.jax import mppi, model, sampler, costs, trajectory, boundary, types
```

## Switching

Replace `trajax.numpy` with `trajax.jax` (or vice versa). All factory functions, types, and return values have the same signatures. With JAX, add a warm-up call before timing-critical code to trigger JIT compilation.

## Consistency

All components in a pipeline must use the same backend. Mixing `trajax.numpy` and `trajax.jax` objects will produce errors.

## JAX Output Conversion

```python
import numpy as np

numpy_array = np.asarray(jax_control.optimal)
```
