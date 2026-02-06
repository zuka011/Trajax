# Notation and Conventions

Quick reference for dimension symbols and array shapes used throughout the API.

## Index Dimensions

| Symbol | Meaning |
|--------|---------|
| $T$ | Time horizon (planning steps) |
| $M$ | Rollout count (samples) |
| $K$ | Obstacle count |
| $N$ | Prediction samples per obstacle |

## State and Input Dimensions

| Symbol | Meaning | Bicycle | Unicycle | Integrator |
|--------|---------|---------|----------|------------|
| $D_x$ | State dimension | 4 $(x, y, \theta, v)$ | 3 $(x, y, \theta)$ | $n$ |
| $D_u$ | Control input dimension | 2 $(a, \delta)$ | 2 $(v, \omega)$ | $n$ |

MPCC adds a virtual component with dimension $D_v = 1$ (path parameter $\phi$) and virtual control $\dot{\phi}$.

## Array Shapes

| Type | Shape | Notes |
|------|-------|-------|
| `State` | $(D_x,)$ | Single state vector |
| `StateSequence` | $(T, D_x)$ | Trajectory over time |
| `StateBatch` | $(T, D_x, M)$ | Batched rollouts |
| `ControlInputSequence` | $(T, D_u)$ | Single control sequence |
| `ControlInputBatch` | $(T, D_u, M)$ | Batched control sequences |
| `PathParameters` | $(T, M)$ | Path progress per rollout |
| `ReferencePoints` | $(T, 3, M)$ | $(x, y, \theta)$ per rollout |
| `ObstacleStates` | $(T, D_o, K)$ | $K$ obstacles over $T$ steps |
| `SampledObstacleStates` | $(T, D_o, K, N)$ | $N$ prediction samples |

## Naming Conventions

| Property | Symbol |
|----------|--------|
| `horizon` | $T$ |
| `dimension` | $D_x$, $D_u$ |
| `rollout_count` | $M$ |
| `count` | $K$ |
| `sample_count` | $N$ |
