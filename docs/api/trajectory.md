# trajectory

Reference trajectories define the path to be followed by MPCC. A trajectory is parameterized by $\phi \in [0, L]$ and can be queried to return position $(x_\phi, y_\phi)$ and heading $\theta_\phi$ at any path parameter value.

## Waypoints Trajectory

Piecewise-linear path through a sequence of 2D waypoints. Headings are computed from segment tangent directions.

::: trajax.trajectories.waypoints.basic.NumPyWaypointsTrajectory
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - query

::: trajax.trajectories.waypoints.accelerated.JaxWaypointsTrajectory
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Line Trajectory

Straight path between two endpoints.

::: trajax.trajectories.line.basic.NumPyLineTrajectory
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - query

::: trajax.trajectories.line.accelerated.JaxLineTrajectory
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Trajectory Protocol

::: trajax.types.Trajectory
    options:
      show_root_heading: true
      heading_level: 3
