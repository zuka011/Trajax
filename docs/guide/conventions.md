# Notation and Conventions

This page documents the notation used throughout trajax for dimensions, shapes, and symbols.

## Dimension Symbols

### Index Dimensions

| Symbol | Description | Example Value |
|--------|-------------|---------------|
| $T$ | Time horizon (number of time steps) | 20 |
| $M$ | Number of rollouts (samples) | 256 |
| $K$ | Number of obstacles | 5 |
| $N$ | Number of prediction samples per obstacle | 10 |

### State/Input Dimensions

| Symbol | Description | Value |
|--------|-------------|-------|
| $D_x$ | State dimension | Model-dependent |
| $D_u$ | Control input dimension | Model-dependent |
| $D_v$ | Virtual state dimension | 2 (MPCC) |
| $D_o$ | Obstacle state dimension | 3 (x, y, heading) |
| $D_r$ | Reference point dimension | 3 (x, y, heading) |

### Bicycle Model Dimensions

| Symbol | Constant | Value | Components |
|--------|----------|-------|------------|
| $D_x$ | `BICYCLE_D_X` | 4 | x, y, heading, speed |
| $D_u$ | `BICYCLE_D_U` | 2 | acceleration, steering |
| $D_v$ | `BICYCLE_D_V` | 2 | path parameter, virtual velocity |
| $D_o$ | `BICYCLE_D_O` | 3 | x, y, heading |

## Array Shapes

### States

| Type | Shape | Description |
|------|-------|-------------|
| State | $(D_x,)$ | Single state |
| StateSequence | $(T, D_x)$ | State trajectory |
| StateBatch | $(T, D_x, M)$ | Batch of rollouts |

### Control Inputs

| Type | Shape | Description |
|------|-------|-------------|
| ControlInputSequence | $(T, D_u)$ | Single input sequence |
| ControlInputBatch | $(T, D_u, M)$ | Batch of input sequences |

### Obstacles

| Type | Shape | Description |
|------|-------|-------------|
| ObstacleStatesForTimeStep | $(D_o, K)$ | Obstacles at one time |
| ObstacleStatesHistory | $(T, D_o, K)$ | Obstacle history |
| SampledObstacleStates | $(T, D_o, K, N)$ | Predicted samples |

### Trajectories

| Type | Shape | Description |
|------|-------|-------------|
| PathParameters | $(T, M)$ | Path progress per rollout |
| Positions | $(T, 2, M)$ | x, y positions |
| ReferencePoints | $(T, D_r, M)$ | x, y, heading |

## Augmented States (MPCC)

MPCC uses augmented states combining physical and virtual components:

```
AugmentedState = Physical + Virtual
```

| Component | Shape | Description |
|-----------|-------|-------------|
| Physical | $(D_x,)$ | Vehicle state (e.g., bicycle) |
| Virtual | $(D_v,)$ | Path parameter, virtual velocity |
| Augmented | $(D_x + D_v,)$ | Combined state |

## Naming Conventions

### Properties

- `horizon` — Time horizon $T$
- `dimension` — State or input dimension ($D_x$, $D_u$, etc.)
- `rollout_count` — Number of samples $M$
- `count` — Number of obstacles $K$
- `sample_count` — Number of prediction samples $N$

### Type Aliases

trajax exports type aliases matching dimension constants:

```python
from trajax.types import BicycleD_x, BicycleD_u, BicycleD_v, D_o, D_r
```

These are parameterized `D[n]` types from numtypes for static shape checking.

## MPPI Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Temperature | $\lambda$ | Softmax sharpness |
| Time step | $\Delta t$ | Discretization interval |
| Horizon | $T$ | Planning steps |
| Samples | $M$ | Rollout count |
