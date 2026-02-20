# types

The types module provides type definitions, protocols, and type aliases used throughout faran.

## Namespace Access

Access type factories through the backend namespaces:

=== "NumPy"

    ```python
    from faran import types

    positions = types.numpy.positions(x=..., y=...)
    ```

=== "JAX"

    ```python
    from faran import types

    positions = types.jax.positions(x=..., y=...)
    ```

## Core Protocols

### State Types

::: faran.types.State
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.StateSequence
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.StateBatch
    options:
      show_root_heading: true
      heading_level: 3

### Control Types

::: faran.types.ControlInputSequence
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.ControlInputBatch
    options:
      show_root_heading: true
      heading_level: 3

### Trajectory Types

- [`Trajectory`](trajectory.md#faran.types.Trajectory) — see [trajectory](trajectory.md)

::: faran.types.PathParameters
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.Positions
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.ReferencePoints
    options:
      show_root_heading: true
      heading_level: 3

### Cost Types

::: faran.types.Costs
    options:
      show_root_heading: true
      heading_level: 3

- [`CostFunction`](costs.md#faran.types.CostFunction) — see [costs](costs.md)

### Boundary Types

::: faran.types.BoundaryDistance
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.BoundaryDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.ExplicitBoundary
    options:
      show_root_heading: true
      heading_level: 3

### Model Types

- [`DynamicalModel`](model.md#faran.types.DynamicalModel) — see [model](model.md)

### MPPI Types

- [`Mppi`](mppi.md#faran.types.Mppi) — see [mppi](mppi.md)

::: faran.types.Sampler
    options:
      show_root_heading: true
      heading_level: 3
