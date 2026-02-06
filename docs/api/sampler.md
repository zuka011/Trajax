# sampler

Samplers generate control input perturbations around a nominal sequence for MPPI rollout exploration.

## Gaussian Sampler

Draws i.i.d. Gaussian perturbations per timestep with a fixed standard deviation per control dimension.

```python
from trajax.numpy import sampler, types
from numtypes import array

control_sampler = sampler.gaussian(
    standard_deviation=array([0.5, 0.2], shape=(2,)),
    rollout_count=256,
    to_batch=types.bicycle.control_input_batch.create,
    seed=42,
)
```

::: trajax.samplers.gaussian.basic.NumPyGaussianSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: trajax.samplers.gaussian.accelerated.JaxGaussianSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Halton Spline Sampler

Generates temporally smooth perturbations using Halton quasi-random sequences interpolated through cubic splines. Halton sequences provide better coverage of the sampling space compared to pseudo-random sampling, and the spline interpolation produces smooth control trajectories.

```python
control_sampler = sampler.halton(
    standard_deviation=array([0.5, 0.2], shape=(2,)),
    rollout_count=256,
    knot_count=8,
    to_batch=types.bicycle.control_input_batch.create,
    seed=42,
)
```

::: trajax.samplers.halton.basic.NumPyHaltonSplineSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

::: trajax.samplers.halton.accelerated.JaxHaltonSplineSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Obstacle State Samplers

Obstacle state samplers draw from predicted obstacle state distributions (Gaussian) for risk-aware collision cost evaluation.

::: trajax.obstacles.sampler.basic.NumPyGaussianObstacle2dPoseSampler
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create

## Sampler Protocol

::: trajax.types.Sampler
    options:
      show_root_heading: true
      heading_level: 3
