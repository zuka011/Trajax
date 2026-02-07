# Obstacle Avoidance

The collision avoidance pipeline has three stages:

```
Obstacle states → Distance computation → Collision cost
```

Optionally, a risk metric replaces the expected collision cost with a risk measure (CVaR, etc.) when obstacle positions are uncertain.

## Distance Computation

Two methods are available for computing signed distances between the ego vehicle and obstacles.

### Circle-to-Circle

Represents both the ego and obstacles as collections of circles. Fast, suitable when precise geometry is not needed.

```python
from trajax.numpy import distance
from trajax import Circles
from numtypes import array

distance_extractor = distance.circles(
    ego=Circles(
        origins=array([[-0.5, 0.0], [0.0, 0.0], [0.5, 0.0]], shape=(3, 2)),
        radii=array([0.8, 0.8, 0.8], shape=(3,)),
    ),
    obstacle=Circles(
        origins=array([[0.0, 0.0]], shape=(1, 2)),
        radii=array([1.0], shape=(1,)),
    ),
    position_extractor=extract.from_physical(position),
    heading_extractor=extract.from_physical(heading),
    obstacle_position_extractor=obstacles.pose_position_extractor,
    obstacle_heading_extractor=obstacles.pose_heading_extractor,
)
```

Distance is center-to-center minus both radii. Negative values indicate overlap.

### SAT (Separating Axis Theorem)

Represents both the ego and obstacles as convex polygons. Computes exact signed separation distance.

```python
from trajax import ConvexPolygon

distance_extractor = distance.sat(
    ego=ConvexPolygon.rectangle(length=2.5, width=1.2),
    obstacle=ConvexPolygon.rectangle(length=2.5, width=1.2),
    position_extractor=extract.from_physical(position),
    heading_extractor=extract.from_physical(heading),
    obstacle_position_extractor=obstacles.pose_position_extractor,
    obstacle_heading_extractor=obstacles.pose_heading_extractor,
)
```

## Obstacle State Provider

The obstacle state provider supplies predicted obstacle positions to the collision cost at each planning step. It wraps a motion predictor and maintains a running history of observations.

```python
from trajax.numpy import obstacles, predictor, model, types

motion_predictor = predictor.curvilinear(
    horizon=30,
    model=model.bicycle.obstacle(time_step_size=0.1, wheelbase=2.5),
    prediction=bicycle_to_obstacle_states,
)

provider = obstacles.provider.predicting(
    predictor=motion_predictor,
    history=types.obstacle_states_running_history.empty(horizon=2),
)

# Feed observations each step
provider.observe(detected_obstacle_states)
```

### Velocity Assumptions

By default the curvilinear predictor assumes **all** estimated velocity components remain constant over the prediction horizon. Sometimes this is too strong — for example, you may want to assume an obstacle drives straight (zero steering) while keeping its current speed.

Pass an `assumptions` callable to override specific components. The callable receives the model-specific velocity object and must return the same type:

```python
from trajax.models.bicycle.basic import NumPyBicycleObstacleVelocities
import numpy as np

# Bicycle: keep speed, zero out steering angle
motion_predictor = predictor.curvilinear(
    horizon=30,
    model=model.bicycle.obstacle(time_step_size=0.1, wheelbase=2.5),
    prediction=bicycle_to_obstacle_states,
    assumptions=lambda v: NumPyBicycleObstacleVelocities(
        steering_angles=np.zeros_like(v.steering_angles),
    ),
)
```

```python
from trajax.models.unicycle.basic import NumPyUnicycleObstacleVelocities

# Unicycle: keep linear velocity, zero out angular velocity
motion_predictor = predictor.curvilinear(
    horizon=30,
    model=model.unicycle.obstacle(time_step_size=0.1),
    prediction=unicycle_to_obstacle_states,
    assumptions=lambda v: NumPyUnicycleObstacleVelocities(
        linear_velocities=v.linear_velocities,
        angular_velocities=np.zeros_like(v.angular_velocities),
    ),
)
```

```python
from trajax.models.integrator.basic import NumPyIntegratorObstacleVelocities

# Integrator: keep first two velocities, zero out the rest
motion_predictor = predictor.curvilinear(
    horizon=30,
    model=model.integrator.obstacle(time_step_size=0.1),
    prediction=integrator_to_obstacle_states,
    assumptions=lambda v: NumPyIntegratorObstacleVelocities(
        np.stack([v.array[0], v.array[1], np.zeros_like(v.array[2])]),
    ),
)
```

The same pattern works with the JAX backend — import the corresponding JAX velocity type instead.

## Obstacle ID Assignment

When obstacles are detected per-frame without persistent IDs, the library can match detections to tracked obstacles across time steps using the Hungarian algorithm on position distances.

```python
id_assignment = obstacles.id_assignment.hungarian(
    position_extractor=obstacle_position_extractor,
    cutoff=5.0,
)
```

The `cutoff` distance (in meters) determines the maximum distance at which a detection can be matched to a tracked obstacle. Detections beyond the cutoff are assigned new IDs. The assignment function is called automatically by the running history when new observations are appended.

### How It Works

1. Computes a pairwise distance matrix between current detections and the last known positions of tracked obstacles
2. Solves the assignment problem via `scipy.optimize.linear_sum_assignment` (NumPy) or an equivalent JAX implementation
3. Pairs with distance ≤ `cutoff` are matched; unmatched detections receive new IDs

Both NumPy and JAX backends are supported. The JAX backend delegates the Hungarian matching to NumPy internally (since the assignment is a small discrete problem) and converts the result back to JAX arrays.

## Collision Cost

The collision cost penalizes rollouts where the signed distance drops below a threshold:

```python
collision = costs.safety.collision(
    obstacle_states=provider,
    sampler=obstacles.sampler.gaussian(seed=44),
    distance=distance_extractor,
    distance_threshold=array([0.5, 0.5, 0.5], shape=(3,)),
    weight=1500.0,
)
```

The `distance_threshold` array has one entry per ego part (e.g., one per circle).

## Risk-Aware Collision Cost

When obstacle positions are uncertain, you can replace the default deterministic evaluation with a risk metric. The collision cost draws $N$ obstacle samples and evaluates the risk measure over the per-sample costs.

```python
from trajax.numpy import risk

collision = costs.safety.collision(
    obstacle_states=provider,
    sampler=obstacles.sampler.gaussian(seed=44),
    distance=distance_extractor,
    distance_threshold=array([0.5, 0.5, 0.5], shape=(3,)),
    weight=1500.0,
    metric=risk.cvar(alpha=0.95, sample_count=50),
)
```

Available risk metrics: `risk.expected_value`, `risk.mean_variance`, `risk.var`, `risk.cvar`, `risk.entropic_risk`. See the [Feature Overview](features.md) for details.
```
