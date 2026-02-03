# Obstacle Avoidance

trajax supports collision avoidance with static and dynamic obstacles.

## Pipeline Overview

```
Observations → Prediction → Distance → Cost
```

## Static Obstacles

```python
from trajax.numpy import obstacles
from numtypes import array
import numpy as np

static_obstacles = obstacles.static(
    positions=array([[15.0, 2.5], [17.0, 20.0]], shape=(2, 2)),
    headings=array([np.pi/6, -np.pi/4], shape=(2,)),
)
```

## Dynamic Obstacles

```python
dynamic_obstacles = obstacles.dynamic(
    positions=array([[25.0, 22.5], [55.0, 0.0]], shape=(2, 2)),
    velocities=array([[0.0, -1.5], [-2.5, 0.0]], shape=(2, 2)),
)

simulator = dynamic_obstacles.with_time_step_size(dt=0.1)
obstacle_states = simulator.step()
```

## Motion Prediction

Predict future positions using curvilinear prediction:

```python
from trajax.numpy import predictor, model

motion_predictor = predictor.curvilinear(
    horizon=30,
    model=model.bicycle.obstacle(time_step_size=0.1, wheelbase=2.5),
    prediction=bicycle_to_obstacle_states,
)
```

## Obstacle State Provider

Manages the prediction pipeline:

```python
from trajax.numpy import obstacles, types

provider = obstacles.provider.predicting(
    predictor=motion_predictor,
    history=types.obstacle_states_running_history.empty(horizon=2),
)

provider.observe(detected_obstacle_states)
predicted_states = provider()
```

## Distance Computation

### Circle-to-Circle

```python
from trajax.numpy import distance
from trajax import Circles

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

Distance is center-to-center minus both radii. Negative values indicate penetration.

### Polygon-to-Polygon

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

## Collision Cost

```python
from trajax.numpy import costs, obstacles

collision = costs.safety.collision(
    obstacle_states=provider,
    sampler=obstacles.sampler.gaussian(seed=44),
    distance=distance_extractor,
    distance_threshold=array([0.5, 0.5, 0.5], shape=(3,)),
    weight=1500.0,
)
```
