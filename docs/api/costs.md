# costs

The costs module provides modular cost function components for trajectory optimization.

## Factory Functions

Access cost factories through the backend namespaces:

=== "NumPy"

    ```python
    from trajax import costs
    
    # Tracking costs
    contouring = costs.numpy.tracking.contouring(...)
    lag = costs.numpy.tracking.lag(...)
    progress = costs.numpy.tracking.progress(...)
    
    # Comfort costs
    smoothing = costs.numpy.comfort.control_smoothing(...)
    effort = costs.numpy.comfort.control_effort(...)
    
    # Safety costs
    collision = costs.numpy.safety.collision(...)
    boundary = costs.numpy.safety.boundary(...)
    
    # Combine costs
    combined = costs.numpy.combined(contouring, lag, collision)
    ```

=== "JAX"

    ```python
    from trajax import costs
    
    # Same API as NumPy
    contouring = costs.jax.tracking.contouring(...)
    combined = costs.jax.combined(...)
    ```

## Tracking Costs

### Contouring Cost

Penalizes lateral deviation from the reference trajectory.

### Lag Cost

Penalizes longitudinal lag behind a reference point.

### Progress Cost

Rewards forward progress along the trajectory.

## Comfort Costs

### Control Smoothing Cost

Penalizes rapid changes in control inputs between time steps.

### Control Effort Cost

Penalizes the magnitude of control inputs.

## Safety Costs

### Collision Cost

Penalizes proximity to obstacles.

### Boundary Cost

Penalizes proximity to or violation of corridor boundaries.

## Combined Costs

The `combined` factory creates a cost function that sums multiple components:

```python
total_cost = costs.numpy.combined(
    costs.numpy.tracking.contouring(...),
    costs.numpy.tracking.lag(...),
    costs.numpy.safety.collision(...),
    costs.numpy.comfort.control_smoothing(...),
)

# Evaluate combined cost
cost_values = total_cost(inputs=control_inputs, states=state_batch)
```

## CostFunction Protocol

All cost components implement the `CostFunction` protocol:

::: trajax.types.CostFunction
    options:
      show_root_heading: true
      heading_level: 3
