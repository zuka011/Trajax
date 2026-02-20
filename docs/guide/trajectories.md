# Defining Trajectories

Trajectories define reference paths for path-following formulations like MPCC.

## Waypoint Trajectories

Piecewise linear path through a sequence of 2D points:

```python
from faran.numpy import trajectory
from numtypes import array

reference = trajectory.waypoints(
    points=array([
        [0.0, 0.0], [10.0, 0.0], [20.0, 5.0], [25.0, 15.0], [20.0, 25.0],
    ], shape=(5, 2)),
    path_length=50.0,
)
```

`path_length` controls how the path is parameterized. If it equals the geometric arc length, the parameterization is "natural" (1 unit of $\phi$ = 1 meter).

## Line Trajectories

Straight path between two endpoints:

```python
reference = trajectory.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0)
```

## Querying

Query a trajectory at specific path parameters to get reference points (position and heading):

```python
from faran.numpy import types

path_params = types.path_parameters(
    array([[0.0, 5.0], [2.5, 7.5]], shape=(2, 2))
)

ref_points = reference.query(path_params)
ref_points.x()        # shape: (T, M)
ref_points.y()        # shape: (T, M)
ref_points.heading()  # shape: (T, M)
```

## Properties

```python
reference.path_length     # user-specified parameterization length
reference.natural_length  # actual geometric arc length
reference.end             # (x, y) of the final point
```

## Cyclic Trajectories

For looped paths (e.g., a race track), enable periodicity in the MPCC config:

```python
planner, model, _, _ = mppi.mpcc(
    ...,
    reference=reference,
    config={"virtual": {"periodic": True}},
)
```
