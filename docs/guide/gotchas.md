# Gotchas

Known limitations and non-obvious behaviors in Faran. If you hit something not listed here, please [open an issue](https://gitlab.com/risk-metrics/faran/-/issues).

## JAX Cost Functions Are Not All Differentiable

Some cost functions implemented in JAX (e.g., collision costs) are not yet automatically differentiable via `jax.grad`. There is no fundamental reason for this, the implementations use operations that could be made differentiable, but until Faran includes a gradient-based planning algorithm, this is not a priority.

**Workaround:** Use the sampling-based [MPPI planner](mppi.md), which does not require gradients.

## SAT Distance Assumes 2D Cartesian Space

The SAT-based distance estimator (`distance.sat`) works with convex polygons in 2D cartesian space. The algorithm can be extended to 3D or non-convex shapes, but this is not currently implemented in Faran.

**Workaround:** Use the circle-based distance estimator (`distance.circles`), which is more general — any shape that can be approximated by a set of circles will work.
