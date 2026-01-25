# Core Concepts

This page explains the fundamental concepts behind trajax.

## Model Predictive Path Integral (MPPI)

MPPI is a sampling-based control algorithm that finds good controls by:

1. **Sampling** candidate control sequences around a nominal trajectory
2. **Simulating** each sequence through a dynamical model
3. **Evaluating** the cost of each resulting trajectory
4. **Weighting** samples by their cost using a softmax function
5. **Averaging** the weighted samples to produce the optimal control

$$
u_{\text{opt}} = \sum_{m=1}^{M} w_m \cdot u_m, \quad w_m = \frac{1}{\eta} \exp\left(-\frac{1}{\lambda}(J_m - J_{\min})\right)
$$

where:

- $M$ is the number of rollouts (samples)
- $\lambda$ is the temperature parameter
- $J_m$ is the cost of rollout $m$

### Temperature Parameter

The temperature $\lambda$ controls exploration vs exploitation:

| Temperature | Behavior |
|-------------|----------|
| Low | Strongly favor lowest-cost samples |
| High | Weight all samples more equally |

## Key Components

### Dynamical Models

A dynamical model defines how your system evolves. The **kinematic bicycle model** is commonly used for wheeled robots:

$$
\dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{\theta} = \frac{v}{L} \tan(\delta), \quad \dot{v} = a
$$

where $L$ is the wheelbase, $\delta$ is steering angle, and $a$ is acceleration.

```python
bicycle = model.bicycle.dynamical(
    time_step_size=0.1,
    wheelbase=2.5,
    speed_limits=(0.0, 15.0),
    steering_limits=(-0.5, 0.5),
    acceleration_limits=(-3.0, 3.0),
)
```

### Samplers

Samplers generate candidate control sequences:

```python
control_sampler = sampler.gaussian(
    standard_deviation=array([0.5, 0.2], shape=(2,)),
    rollout_count=256,
    to_batch=types.bicycle.control_input_batch.create,
    seed=42,
)
```

A Halton sampler provides more uniform coverage:

```python
halton_sampler = sampler.halton(
    standard_deviation=array([0.5, 0.2], shape=(2,)),
    rollout_count=256,
    knot_count=6,
    to_batch=types.bicycle.control_input_batch.create,
    seed=42,
)
```

### Cost Functions

Cost functions evaluate trajectory quality:

**Tracking:**

- `costs.tracking.contouring` — Lateral deviation from path
- `costs.tracking.lag` — Longitudinal deviation from reference point  
- `costs.tracking.progress` — Reward forward motion

**Safety:**

- `costs.safety.collision` — Penalize obstacle proximity
- `costs.safety.boundary` — Penalize leaving corridor

**Comfort:**

- `costs.comfort.control_smoothing` — Penalize control rate of change
- `costs.comfort.control_effort` — Penalize control magnitude

Combine them:

```python
cost = costs.combined(
    costs.tracking.contouring(reference=reference, weight=50.0, ...),
    costs.tracking.lag(reference=reference, weight=100.0, ...),
    costs.safety.collision(obstacle_states=..., weight=1500.0, ...),
)
```

## MPCC: Model Predictive Contouring Control

MPCC is a path-following formulation. It introduces a **virtual path parameter** $\phi$ that tracks progress along the reference trajectory.

### Contouring/Lag Decomposition

MPCC splits tracking error into:

- **Contouring error** $e_c$: Perpendicular distance to the path
- **Lag error** $e_l$: Parallel distance behind the reference point

$$
e_c = \sin(\theta_\phi)(x - x_\phi) - \cos(\theta_\phi)(y - y_\phi)
$$

$$
e_l = -\cos(\theta_\phi)(x - x_\phi) - \sin(\theta_\phi)(y - y_\phi)
$$

The progress cost pushes $\phi$ forward while contouring and lag costs pull it back, creating stable tracking.

### MPCC Factory

```python
planner, augmented_model, contouring_cost, lag_cost = mppi.mpcc(
    model=bicycle,
    sampler=control_sampler,
    reference=reference,
    position_extractor=extract.from_physical(position),
    config={
        "weights": {"contouring": 50.0, "lag": 100.0, "progress": 1000.0},
        "virtual": {"velocity_limits": (0.0, 15.0)},
    },
)
```

## State Representations

### Batched Operations

Computations operate on **batches** with shape `(T, D_x, M)`:

- `T`: Time horizon
- `D_x`: State dimension
- `M`: Number of rollouts

### Augmented States

For MPCC, states combine physical and virtual components:

```python
state = types.augmented.state.of(
    physical=types.bicycle.state.create(x=0.0, y=0.0, heading=0.0, speed=0.0),
    virtual=types.simple.state.zeroes(dimension=1),  # path parameter
)
```

## Extractors

Extractors decouple state representations from cost computations:

```python
# For augmented states
position_extractor = extract.from_physical(
    lambda states: types.positions(x=states.positions.x(), y=states.positions.y())
)
```

## Control Flow

Each planning step:

1. `sampler.sample()` — Generate control samples
2. `model.simulate()` — Rollout each sample
3. `cost()` — Evaluate costs
4. Compute softmax weights
5. `weighted_sum()` — Combine samples
6. `shift_and_pad()` — Prepare for next timestep
