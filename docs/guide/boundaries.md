# Boundaries

Boundaries define drivable corridors around a reference trajectory. They produce signed distances that feed into a boundary cost.

## Fixed-Width Corridor

Constant width on each side of the reference path:

```python
from faran.numpy import boundary, trajectory, types

reference = trajectory.line(start=(0.0, 0.0), end=(10.0, 0.0), path_length=10.0)

corridor = boundary.fixed_width(
    reference=reference,
    position_extractor=lambda states: types.positions(
        x=states.array[:, 0, :], y=states.array[:, 1, :],
    ),
    left=2.0,
    right=5.0,
)
```

## Variable-Width Corridor

Width changes at arc-length breakpoints:

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

## Distance Convention

Calling the corridor returns signed distances with shape $(T, M)$:

```python
distances = corridor(states=states)
```

| Value | Meaning |
|-------|---------|
| $d > 0$ | Inside corridor |
| $d = 0$ | On boundary |
| $d < 0$ | Violation |

## Boundary Cost

```python
from faran.numpy import costs

boundary_cost = costs.safety.boundary(
    distance=corridor,
    distance_threshold=0.25,
    weight=1000.0,
)
```

The cost activates when the signed distance drops below `distance_threshold`, penalizing states that approach or cross the boundary.
