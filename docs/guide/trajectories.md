# Defining Trajectories

Trajectories define reference paths for path-following control.

## Line Trajectories

A line trajectory is a straight path between two points:

```python
from trajax.numpy import trajectory

reference = trajectory.line(
    start=(0.0, 0.0),
    end=(10.0, 0.0),
    path_length=10.0,
)
```

The `path_length` controls how the path is parameterized. If it equals the Euclidean distance, the parameterization is "natural" (1 unit = 1 meter).

## Waypoint Trajectories

For curved paths:

```python
from numtypes import array

reference = trajectory.waypoints(
    points=array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 5.0],
        [25.0, 15.0],
        [20.0, 25.0],
    ], shape=(5, 2)),
    path_length=50.0,
)
```

Waypoint trajectories interpolate linearly between points. Headings are computed from the path tangent direction.

## Path Properties

```python
end_x, end_y = reference.end
total_length = reference.path_length     # User-specified arc length
natural_length = reference.natural_length  # Actual geometric length
```

## Querying Trajectories

```python
from trajax.numpy import types
from numtypes import array

# Query at multiple path parameters (shape: T, M)
path_params = types.path_parameters(
    array([[0.0, 5.0], [2.5, 7.5], [5.0, 10.0]], shape=(3, 2))
)

ref_points = reference.query(path_params)

x_positions = ref_points.x      # shape: (T, M)
y_positions = ref_points.y      # shape: (T, M)
headings = ref_points.heading   # shape: (T, M)
```

## Using in MPCC

```python
from trajax.numpy import trajectory, mppi, model, sampler, types
from trajax.states import extract
from numtypes import array

def position(states):
    return types.positions(x=states.positions.x(), y=states.positions.y())

reference = trajectory.waypoints(
    points=array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 5.0],
        [25.0, 15.0],
        [20.0, 25.0],
    ], shape=(5, 2)),
    path_length=50.0,
)

planner, model, contouring, lag = mppi.mpcc(
    model=model.bicycle.dynamical(...),
    sampler=sampler.gaussian(...),
    reference=reference,
    position_extractor=extract.from_physical(position),
)
```

## Cyclic Trajectories

For trajectories that loop (e.g., a race track):

```python
planner, model, contouring, lag = mppi.mpcc(
    ...
    reference=reference,
    config={
        "virtual": {"periodic": True},
    },
)
```

## Custom Trajectories

Implement the `Trajectory` protocol:

```python
from trajax import Trajectory, PathParameters, ReferencePoints

class MyTrajectory(Trajectory):
    @property
    def path_length(self) -> float:
        return self._length
    
    @property
    def natural_length(self) -> float:
        return self._natural_length
    
    @property
    def end(self) -> tuple[float, float]:
        return self._end
    
    def query(self, path_parameters: PathParameters) -> ReferencePoints:
        ...
```
