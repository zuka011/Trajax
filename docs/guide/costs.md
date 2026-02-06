# Cost Function Design

A cost function receives batched states and control inputs and returns a cost per rollout per time step. The MPPI algorithm sums costs over time, then uses softmax weighting to combine rollouts. Lower cost is better.

```python
costs_array = cost_function(inputs=control_batch, states=state_batch)
# costs_array.array has shape (T, M)
```

## Tracking Costs

These are used in the MPCC formulation (see [Concepts](concepts.md#mpcc-model-predictive-contouring-control)).

### Contouring

Penalizes lateral deviation from the reference path:

$$
J_c = k_c \cdot e_c^2
$$

```python
contouring = costs.tracking.contouring(
    reference=reference,
    path_parameter_extractor=extract.from_virtual(path_parameter),
    position_extractor=extract.from_physical(position),
    weight=50.0,
)
```

### Lag

Penalizes longitudinal deviation from the reference point:

$$
J_l = k_l \cdot e_l^2
$$

```python
lag = costs.tracking.lag(
    reference=reference,
    path_parameter_extractor=extract.from_virtual(path_parameter),
    position_extractor=extract.from_physical(position),
    weight=100.0,
)
```

### Progress

Rewards forward motion along the path. Without this, $\phi$ would stay at zero:

$$
J_p = -k_p \cdot \dot\phi \cdot \Delta t
$$

```python
progress = costs.tracking.progress(
    path_velocity_extractor=extract.from_virtual(path_velocity),
    time_step_size=0.1,
    weight=1000.0,
)
```

## Safety Costs

### Collision

Penalizes proximity to obstacles when the signed distance falls below a threshold:

$$
J_{\text{col}} = k_{\text{col}} \max(d_0 - d, \; 0)
$$

See [Obstacle Handling](obstacles.md) for how to set up obstacle states, distance extractors, and samplers.

```python
collision = costs.safety.collision(
    obstacle_states=provider,
    sampler=obstacles.sampler.gaussian(seed=44),
    distance=distance_extractor,
    distance_threshold=array([0.5, 0.5, 0.5], shape=(3,)),
    weight=1500.0,
)
```

### Boundary

Penalizes states approaching the edges of a corridor. See [Boundaries](boundaries.md).

```python
boundary_cost = costs.safety.boundary(
    distance=corridor,
    distance_threshold=0.25,
    weight=1000.0,
)
```

## Comfort Costs

### Control Smoothing

Penalizes the difference between consecutive control inputs:

```python
smoothing = costs.comfort.control_smoothing(
    weights=array([5.0, 20.0, 5.0], shape=(3,)),
)
```

### Control Effort

Penalizes the magnitude of control inputs:

```python
effort = costs.comfort.control_effort(
    weights=array([0.1, 0.5, 0.1], shape=(3,)),
)
```

## Combining Costs

`costs.combined` sums any number of cost components:

```python
total = costs.combined(
    costs.tracking.contouring(...),
    costs.tracking.lag(...),
    costs.tracking.progress(...),
    costs.safety.collision(...),
    costs.comfort.control_smoothing(...),
)
```

## Custom Cost Functions

Any callable with the right signature works as a cost function. It must accept keyword arguments `inputs` and `states` and return a costs object:

```python
import numpy as np

def speed_limit_cost(*, inputs, states, target_speed=5.0, weight=10.0):
    speeds = states.array[:, 3, :]
    return types.numpy.costs(weight * (speeds - target_speed) ** 2)

total = costs.combined(
    costs.tracking.contouring(...),
    speed_limit_cost,
)
```
