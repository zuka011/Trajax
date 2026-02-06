# mppi

MPPI (Model Predictive Path Integral) is a sampling-based stochastic optimal control algorithm [^1] [^2]. At each planning step, it samples $M$ control input perturbations around a nominal sequence, simulates rollouts through a dynamical model, evaluates a cost function, and computes a cost-weighted average as the optimal control:

$$
\mathbf{u}_{\text{opt}} = \sum_{m=1}^{M} w_m \, \mathbf{u}_m, \quad w_m = \frac{1}{\eta} \exp\!\left(\frac{-1}{\lambda}(J_m - J_{\min})\right)
$$

where $\lambda$ is the temperature parameter controlling the sharpness of the softmax weighting.

[^1]: G. Williams, A. Aldrich, E. Theodorou, "Model Predictive Path Integral Control using Covariance Variable Importance Sampling," arXiv:1509.01149, 2015.
[^2]: G. Williams et al., "Aggressive Driving with Model Predictive Path Integral Control," IEEE ICRA, 2016.

## Factory Functions

```python
from trajax.numpy import mppi

# Base MPPI planner
planner = mppi.base(model=..., cost_function=..., sampler=...)

# MPPI with augmented state (physical + virtual)
planner = mppi.augmented(physical=..., virtual=..., ...)

# MPCC convenience factory (assembles augmented model, contouring/lag/progress costs)
planner, augmented_model, contouring, lag = mppi.mpcc(model=..., sampler=..., reference=..., ...)
```

## NumPyMppi

::: trajax.mppi.basic.NumPyMppi
    options:
      show_root_heading: true
      heading_level: 3
      show_source: true
      members:
        - create
        - step

## JaxMppi

::: trajax.mppi.accelerated.JaxMppi
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - step

## MPCC Factory

::: trajax.mpcc.basic.NumPyMpccMppi
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: trajax.mpcc.accelerated.JaxMpccMppi
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Mppi Protocol

::: trajax.types.Mppi
    options:
      show_root_heading: true
      heading_level: 3

## Supporting Types

### Control

::: trajax.types.Control
    options:
      show_root_heading: true
      heading_level: 4

### Update Functions

::: trajax.mppi.common.UseOptimalControlUpdate
    options:
      show_root_heading: true
      heading_level: 4

::: trajax.mppi.common.NoFilter
    options:
      show_root_heading: true
      heading_level: 4

### Savitzky-Golay Filter

::: trajax.mppi.savgol.basic.NumPySavGolFilter
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

::: trajax.mppi.savgol.accelerated.JaxSavGolFilter
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create
