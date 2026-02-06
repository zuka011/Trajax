# Getting Started

## Installation

```bash
pip install trajax
```

Requires Python 3.13+.

## Minimal MPCC Example

The fastest way to a working planner is `mppi.mpcc()`, which assembles an MPPI planner with contouring, lag, and progress costs for path following:

```python
from trajax.numpy import mppi, model, sampler, trajectory, types, extract
from numtypes import array

def position(states):
    return types.positions(x=states.positions.x(), y=states.positions.y())

reference = trajectory.waypoints(
    points=array([
        [0.0, 0.0], [10.0, 0.0], [20.0, 5.0], [25.0, 15.0], [20.0, 25.0],
    ], shape=(5, 2)),
    path_length=50.0,
)

planner, augmented_model, contouring, lag = mppi.mpcc(
    model=model.bicycle.dynamical(
        time_step_size=0.1,
        wheelbase=2.5,
        speed_limits=(0.0, 15.0),
        steering_limits=(-0.5, 0.5),
        acceleration_limits=(-3.0, 3.0),
    ),
    sampler=sampler.gaussian(
        standard_deviation=array([0.5, 0.2], shape=(2,)),
        rollout_count=256,
        to_batch=types.bicycle.control_input_batch.create,
        seed=42,
    ),
    reference=reference,
    position_extractor=extract.from_physical(position),
    config={
        "weights": {"contouring": 50.0, "lag": 100.0, "progress": 1000.0},
        "virtual": {"velocity_limits": (0.0, 15.0)},
    },
)
```

## Planning Loop

```python
current_state = types.augmented.state.of(
    physical=types.bicycle.state.create(x=0.0, y=0.0, heading=0.0, speed=0.0),
    virtual=types.simple.state.zeroes(dimension=1),
)
nominal = types.augmented.control_input_sequence.of(
    physical=types.bicycle.control_input_sequence.zeroes(horizon=30),
    virtual=types.simple.control_input_sequence.zeroes(horizon=30, dimension=1),
)

for step in range(150):
    control = planner.step(
        temperature=50.0,
        nominal_input=nominal,
        initial_state=current_state,
    )
    nominal = control.nominal
    current_state = augmented_model.step(inputs=control.optimal, state=current_state)

    if current_state.virtual.array[0] >= reference.path_length * 0.9:
        break
```

## What `mppi.mpcc()` Sets Up

MPCC augments the vehicle state with a virtual path parameter $\phi$:

| Component | State | Controls |
|-----------|-------|----------|
| Physical | $[x, y, \theta, v]$ | $[a, \delta]$ |
| Virtual | $[\phi]$ | $[\dot{\phi}]$ |

Three costs drive path following:

- **Contouring** — penalizes lateral deviation from the reference
- **Lag** — penalizes longitudinal offset between $\phi$ and the vehicle's projection
- **Progress** — rewards forward motion along the path

For full manual assembly (custom models, additional costs, mixed samplers), see [Core Concepts](concepts.md).

## Next Steps

- [Core Concepts](concepts.md) — MPPI algorithm and MPCC formulation
- [MPPI Planning](mppi.md) — Temperature, filtering, seeding
- [Costs](costs.md) — Tracking, safety, and comfort objectives
- [Obstacles](obstacles.md) — Collision avoidance with distance functions
