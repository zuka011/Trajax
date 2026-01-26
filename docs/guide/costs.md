# Cost Function Design

Cost functions define what behaviors the MPPI planner should optimize for.

## How Costs Work

A cost function takes batched states and control inputs, and returns a cost for each rollout:

```python
costs = cost_function(inputs=control_batch, states=state_batch)
```

Lower costs are better. The MPPI algorithm weights samples by their total cost.

## Tracking Costs

### Contouring Cost

Penalizes **lateral deviation** from the reference path:

$$
J_c = k_c \cdot e_c^2, \quad e_c = \sin(\theta_\phi)(x - x_\phi) - \cos(\theta_\phi)(y - y_\phi)
$$

```python
from trajax.numpy import costs, trajectory, types, extract
from numtypes import array

def position(states):
    return types.positions(x=states.positions.x(), y=states.positions.y())

def path_parameter(states):
    return types.path_parameters(states.array[:, 0, :])

contouring = costs.tracking.contouring(
    reference=reference,
    path_parameter_extractor=extract.from_virtual(path_parameter),
    position_extractor=extract.from_physical(position),
    weight=50.0,
)
```

### Lag Cost

Penalizes **longitudinal deviation** from the reference point:

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

### Progress Cost

**Rewards forward motion** by penalizing negative path velocity:

$$
J_p = -k_p \cdot \dot{\phi} \cdot \Delta t
$$

```python
def path_velocity(inputs):
    return inputs.array[:, 0, :]

progress = costs.tracking.progress(
    path_velocity_extractor=extract.from_virtual(path_velocity),
    time_step_size=0.1,
    weight=1000.0,
)
```

## Safety Costs

### Collision Cost

Penalizes proximity to obstacles:

$$
J_{\text{col}} = \begin{cases}
k_{\text{col}} (d_0 - d) & \text{if } d < d_0 \\
0 & \text{otherwise}
\end{cases}
$$

```python
from trajax.numpy import costs, distance, obstacles
from trajax import Circles

collision = costs.safety.collision(
    obstacle_states=obstacle_provider,
    sampler=obstacles.sampler.gaussian(seed=44),
    distance=distance.circles(ego=ego, obstacle=obstacle, ...),
    distance_threshold=array([0.5, 0.5, 0.5], shape=(3,)),
    weight=1500.0,
)
```

### Boundary Cost

Penalizes leaving a corridor:

```python
from trajax.numpy import boundary

corridor = boundary.fixed_width(
    reference=reference,
    position_extractor=extract.from_physical(position),
    left=2.5,
    right=2.5,
)

boundary_cost = costs.safety.boundary(
    distance=corridor,
    distance_threshold=0.25,
    weight=1000.0,
)
```

## Comfort Costs

### Control Smoothing

Penalizes the rate of change of control inputs:

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

```python
cost = costs.combined(
    costs.tracking.contouring(weight=50.0, ...),
    costs.tracking.lag(weight=100.0, ...),
    costs.tracking.progress(weight=1000.0, ...),
    costs.safety.collision(weight=1500.0, ...),
)
```

## Custom Cost Functions

Any callable with the right signature works:

```python
def speed_limit_cost(inputs, states, *, target_speed=5.0, weight=10.0):
    speeds = states.array[:, 3, :]
    error = (speeds - target_speed) ** 2
    return types.numpy.costs(weight * np.sum(error, axis=0))

cost = costs.combined(
    costs.tracking.contouring(...),
    speed_limit_cost,
)
```
