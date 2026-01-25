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

::: trajax.types.Trajectory
    options:
      show_root_heading: true
      heading_level: 3

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

::: trajax.types.CostFunction
    options:
      show_root_heading: true
      heading_level: 3

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

::: trajax.types.DynamicalModel
    options:
      show_root_heading: true
      heading_level: 3

### MPPI Types

::: trajax.types.Mppi
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.Sampler
    options:
      show_root_heading: true
      heading_level: 3
