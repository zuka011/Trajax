# boundary

Boundaries define drivable corridors around a reference trajectory and compute signed distances from vehicle positions to corridor edges.

## Fixed-Width Boundary

Constant corridor width on each side of the reference path.

::: faran.costs.boundary.basic.NumPyFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.costs.boundary.accelerated.JaxFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Piecewise Fixed-Width Boundary

Corridor width varies at arc-length breakpoints along the reference.

::: faran.costs.boundary.basic.NumPyPiecewiseFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: faran.costs.boundary.accelerated.JaxPiecewiseFixedWidthBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Boundary Protocols

::: faran.types.BoundaryDistanceExtractor
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.ExplicitBoundary
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - left
        - right

## Width Types

::: faran.types.BoundaryWidths
    options:
      show_root_heading: true
      heading_level: 3

::: faran.types.BoundaryWidthsDescription
    options:
      show_root_heading: true
      heading_level: 3
