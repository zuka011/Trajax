# boundary

Boundaries define drivable corridors around a reference trajectory and compute signed distances from vehicle positions to corridor edges.

## Fixed-Width Boundary

Constant corridor width on each side of the reference path.

::: trajax.costs.boundary.basic.NumPyFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: trajax.costs.boundary.accelerated.JaxFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Piecewise Fixed-Width Boundary

Corridor width varies at arc-length breakpoints along the reference.

::: trajax.costs.boundary.basic.NumPyPiecewiseFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: trajax.costs.boundary.accelerated.JaxPiecewiseFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Boundary Protocols

::: trajax.types.BoundaryDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.ExplicitBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - left
        - right

## Width Types

::: trajax.types.BoundaryWidths
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.types.BoundaryWidthsDescription
    options:
      show_root_heading: true
      heading_level: 3
