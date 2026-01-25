# boundary

The boundary module provides corridor boundary extractors for constraint checking.

## Factory Functions

Access boundary factories through the backend namespaces:

=== "NumPy"

    ```python
    from trajax import boundary
    
    corridor = boundary.numpy.fixed_width(...)
    corridor = boundary.numpy.piecewise_fixed_width(...)
    ```

=== "JAX"

    ```python
    from trajax import boundary
    
    corridor = boundary.jax.fixed_width(...)
    corridor = boundary.jax.piecewise_fixed_width(...)
    ```

## Fixed-Width Boundary

Creates a corridor with constant width along the entire reference trajectory.

::: trajax.costs.boundary.basic.NumPyFixedWidthBoundary.create
    options:
      show_root_heading: true
      heading_level: 3

## Piecewise Fixed-Width Boundary

Creates a corridor where width varies by longitudinal segment.

::: trajax.costs.boundary.basic.NumPyPiecewiseFixedWidthBoundary.create
    options:
      show_root_heading: true
      heading_level: 3

## Width Specification Types

::: trajax.types.BoundaryWidths
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.BoundaryWidthsDescription
    options:
      show_root_heading: true
      heading_level: 3

## Boundary Protocols

### BoundaryDistanceExtractor

Callable protocol for computing signed distances to boundaries:

::: trajax.types.BoundaryDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 3

### ExplicitBoundary

Protocol for boundaries that can sample explicit boundary points:

::: trajax.types.ExplicitBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - left
        - right
