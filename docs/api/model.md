# model

The model module provides dynamical models for vehicle simulation and state propagation.

## Factory Functions

Access model factories through the backend namespaces:

=== "NumPy"

    ```python
    from trajax import model
    
    # Kinematic bicycle model
    bicycle = model.numpy.bicycle.dynamical(time_step_size=0.1)
    ```

=== "JAX"

    ```python
    from trajax import model
    
    # Kinematic bicycle model with JIT compilation
    bicycle = model.jax.bicycle.dynamical(time_step_size=0.1)
    ```

## Bicycle Model

The kinematic bicycle model represents vehicle dynamics with four state variables:

- `x`: x-position
- `y`: y-position  
- `heading`: vehicle orientation (radians)
- `speed`: forward velocity

Control inputs are:

- `acceleration`: longitudinal acceleration
- `steering`: front wheel steering angle

### Factory Function

::: trajax.models.bicycle.basic.NumPyBicycleModel.create
    options:
      show_root_heading: true
      heading_level: 3

## DynamicalModel Protocol

All dynamical models implement the [`DynamicalModel`](types.md#trajax.types.DynamicalModel) protocol, which provides:

- `simulate(inputs, initial_state)`: Simulate vehicle trajectory given control inputs
