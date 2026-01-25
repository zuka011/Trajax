# Working with Boundaries

Boundaries define drivable corridors around reference trajectories.

## Fixed-Width Boundaries

```python
from trajax.numpy import trajectory, boundary, types

reference = trajectory.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0)

corridor = boundary.fixed_width(
    reference=reference,
    position_extractor=lambda states: types.positions(
        x=states.array[:, 0, :],
        y=states.array[:, 1, :],
    ),
    left=2.0,
    right=5.0,
)
```

## Piecewise Fixed-Width Boundaries

For corridors that change width along the path:

```python
corridor = boundary.piecewise_fixed_width(
    reference=reference,
    position_extractor=position_extractor,
    widths={
        0.0: {"left": 2.0, "right": 4.0},
        5.0: {"left": 3.0, "right": 5.0},
        7.0: {"left": 1.0, "right": 2.0},
    },
)
```

Each key is an arc-length breakpoint.

## Computing Boundary Distances

```python
distances = corridor(states=states)  # Shape: (T, M)
```

### Distance Sign Convention

| Value | Meaning |
|-------|---------|
| `distance > 0` | Inside corridor |
| `distance = 0` | On boundary |
| `distance < 0` | Outside (violation) |

## Boundary Cost

```python
from trajax.numpy import costs

boundary_cost = costs.safety.boundary(
    distance=corridor,
    distance_threshold=0.25,
    weight=1000.0,
)
```

## Position Extractors

```python
# For simple states
position_extractor = lambda states: types.positions(
    x=states.array[:, 0, :],
    y=states.array[:, 1, :],
)

# For augmented states
from trajax.states import extract

position_extractor = extract.from_physical(
    lambda s: types.positions(x=s.positions.x(), y=s.positions.y())
)
```
