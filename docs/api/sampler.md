# sampler

The sampler module provides control input samplers for MPPI and obstacle state samplers for risk-aware planning.

## Control Input Samplers

=== "NumPy"

    ```python
    from trajax import sampler
    
    # Gaussian sampler with fixed seed for reproducibility
    control_sampler = sampler.numpy.gaussian(seed=42)
    
    # Sample control inputs around a nominal trajectory
    sampled_inputs = control_sampler(nominal=nominal_inputs)
    ```

=== "JAX"

    ```python
    from trajax import sampler
    
    # JIT-compiled Gaussian sampler
    control_sampler = sampler.jax.gaussian(seed=42)
    ```

## Obstacle State Samplers

Used for sampling from predicted obstacle distributions:

=== "NumPy"

    ```python
    from trajax import obstacles
    
    # Gaussian sampler for obstacle predictions
    obstacle_sampler = obstacles.numpy.sampler.gaussian(seed=42)
    
    # Sample N possible obstacle positions from predictions
    samples = obstacle_sampler(obstacle_states, count=N)
    ```

=== "JAX"

    ```python
    from trajax import obstacles
    
    obstacle_sampler = obstacles.jax.sampler.gaussian(seed=42)
    ```

## Sampler Protocol

All samplers implement the [`Sampler`](types.md#trajax.types.Sampler) protocol.

## Obstacle State Sampler Protocol

Obstacle state samplers implement the `ObstacleStateSampler` protocol.
