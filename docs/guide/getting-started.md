# Getting Started

## Installation

```bash
pip install trajax
```

Requires Python 3.13+.

## Minimal MPCC Example

The fastest way to a working planner is `mppi.mpcc()`, which assembles an MPPI planner with contouring, lag, and progress costs for path following:

```python
--8<-- "docs/examples/01_basic_path_following.py:setup"
```

## Planning Loop

```python
--8<-- "docs/examples/01_basic_path_following.py:loop"
```

??? note "Full example"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py"
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
