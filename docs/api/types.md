# types

The types module provides type definitions, protocols, and type aliases used throughout trajax.

## Namespace Access

Access type factories through the backend namespaces:

=== "NumPy"

    ```python
    from trajax import types

    positions = types.numpy.positions(x=..., y=...)
    ```

=== "JAX"

    ```python
    from trajax import types

    positions = types.jax.positions(x=..., y=...)
    ```

## Core Protocols

### State Types

::: trajax.types.State
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.StateSequence
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.StateBatch
    options:
      show_root_heading: true
      heading_level: 3

### Control Types

::: trajax.types.ControlInputSequence
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.ControlInputBatch
    options:
      show_root_heading: true
      heading_level: 3

### Trajectory Types

- [`Trajectory`](trajectory.md#trajax.types.Trajectory) — see [trajectory](trajectory.md)

::: trajax.types.PathParameters
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.Positions
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.ReferencePoints
    options:
      show_root_heading: true
      heading_level: 3

### Cost Types

::: trajax.types.Costs
    options:
      show_root_heading: true
      heading_level: 3

- [`CostFunction`](costs.md#trajax.types.CostFunction) — see [costs](costs.md)

### Boundary Types

::: trajax.types.BoundaryDistance
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.BoundaryDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.ExplicitBoundary
    options:
      show_root_heading: true
      heading_level: 3

### Model Types

- [`DynamicalModel`](model.md#trajax.types.DynamicalModel) — see [model](model.md)

### MPPI Types

- [`Mppi`](mppi.md#trajax.types.Mppi) — see [mppi](mppi.md)

::: trajax.types.Sampler
    options:
      show_root_heading: true
      heading_level: 3
