# trajectory

The trajectory module provides reference path definitions for trajectory planning.

## Factory Functions

Access trajectory factories through the backend namespaces:

=== "NumPy"

    ```python
    from trajax import trajectory
    
    reference = trajectory.numpy.line(...)
    reference = trajectory.numpy.waypoints(...)
    ```

=== "JAX"

    ```python
    from trajax import trajectory
    
    reference = trajectory.jax.line(...)
    reference = trajectory.jax.waypoints(...)
    ```

## Line Trajectory

Creates a straight-line trajectory between two points.

::: trajax.trajectories.line.basic.NumPyLineTrajectory.create
    options:
      show_root_heading: true
      heading_level: 3

## Waypoints Trajectory

Creates a piecewise-linear trajectory through waypoints.

::: trajax.trajectories.waypoints.basic.NumPyWaypointsTrajectory.create
    options:
      show_root_heading: true
      heading_level: 3

## Trajectory Protocol

All trajectories implement the [`Trajectory`](types.md#trajax.types.Trajectory) protocol, which defines methods for querying positions, computing lateral/longitudinal errors, and getting path length.
