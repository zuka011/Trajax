# mppi

The mppi module provides Model Predictive Path Integral (MPPI) controllers for trajectory optimization.

## Factory Functions

Access MPPI factories through the backend namespaces:

=== "NumPy"

    ```python
    from trajax import mppi, model, sampler, costs
    
    controller = mppi.numpy.base(
        model=model.numpy.bicycle.dynamical(time_step_size=0.1),
        cost_function=costs.numpy.tracking(reference=..., ...),
        sampler=sampler.numpy.gaussian(seed=42),
    )
    ```

=== "JAX"

    ```python
    from trajax import mppi, model, sampler, costs
    
    controller = mppi.jax.base(
        model=model.jax.bicycle.dynamical(time_step_size=0.1),
        cost_function=costs.jax.tracking(reference=..., ...),
        sampler=sampler.jax.gaussian(seed=42),
    )
    ```

## Controller Classes

::: trajax.mppi.basic.NumPyMppi
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __call__

::: trajax.mppi.accelerated.JaxMppi
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - __call__

## Mppi Protocol

All MPPI controllers implement the [`Mppi`](types.md#trajax.types.Mppi) protocol.
