# Core Concepts

## MPPI

MPPI (Model Predictive Path Integral) is a sampling-based trajectory optimization algorithm. At each planning step it:

1. Draws $M$ control sequences by perturbing a nominal sequence
2. Simulates each through a dynamics model to produce $M$ state rollouts
3. Evaluates a cost function on every rollout
4. Computes a softmax-weighted average of the samples as the optimal control

$$
u^* = \sum_{m=1}^{M} w_m \, u_m, \quad w_m = \frac{1}{\eta} \exp\!\left(-\frac{1}{\lambda}\,(J_m - J_{\min})\right)
$$

The temperature $\lambda$ controls how aggressively the planner favors low-cost samples. Low temperature concentrates weight on the best samples; high temperature distributes weight more evenly.

### What You Provide

| Component | Role |
|---|---|
| Dynamics model | Predicts the next state given current state and control input |
| Cost function | Scores each rollout (lower is better) |
| Sampler | Generates perturbations around the nominal control sequence |

faran provides concrete implementations for each of these (see [Feature Overview](features.md)), and you compose them into a planner via factory functions.

### Factory Functions

| Factory | When to use |
|---|---|
| `mppi.base` | You have a single model and a custom cost function |
| `mppi.augmented` | Your state has multiple components (e.g., physical + virtual) |
| `mppi.mpcc` | Path following with the MPCC formulation (see below) |

```python
from faran.numpy import mppi

# Lowest-level: bring your own model, cost, sampler
planner = mppi.base(model=..., cost_function=..., sampler=...)

# Highest-level: MPCC path following in one call
planner, model, contouring, lag = mppi.mpcc(model=..., sampler=..., reference=..., ...)
```

## MPCC: Model Predictive Contouring Control

MPCC is an MPC formulation for path following. It introduces a virtual path parameter $\phi$ that moves independently along the reference trajectory, and decomposes tracking error into two components:

- **Contouring error** $e_c$ — perpendicular distance to the path (lateral deviation)
- **Lag error** $e_l$ — distance along the path behind the reference point (longitudinal deviation)

$$
e_c = \sin(\theta_\phi)(x - x_\phi) - \cos(\theta_\phi)(y - y_\phi)
$$

$$
e_l = -\cos(\theta_\phi)(x - x_\phi) - \sin(\theta_\phi)(y - y_\phi)
$$

A progress cost pushes $\phi$ forward while contouring and lag costs pull the vehicle toward the reference point. The balance between these three costs determines tracking behavior.

### Augmented State

MPCC augments the physical state with a virtual component:

| | Variables | Meaning |
|---|---|---|
| Physical state | $x, y, \theta, v$ | Vehicle pose and speed |
| Virtual state | $\phi$ | Arc-length progress along the reference |
| Physical controls | $a, \delta$ | Acceleration, steering |
| Virtual control | $\dot\phi$ | Path velocity |

Both the physical and virtual dynamics are simulated together. The `mppi.mpcc` factory handles this composition automatically.

## Batched Computation

All computations operate on 3D tensors:

| Shape | Meaning |
|---|---|
| $(T, D_x, M)$ | State batch — $T$ time steps, $D_x$ state dimensions, $M$ rollouts |
| $(T, D_u, M)$ | Control batch — same layout for control inputs |
| $(T, M)$ | Cost array — one scalar per rollout per time step |

The MPPI algorithm sums costs over $T$ to get a total cost per rollout, then computes softmax weights to combine all $M$ samples.

## Extractors

Extractors decouple cost functions from specific state representations. A cost function never accesses state arrays directly — it asks an extractor for the values it needs (positions, headings, path parameters).

This lets you reuse the same cost function implementation with different models:

```python
from faran.numpy import extract, types

# For augmented states: extract position from the physical sub-state
position_extractor = extract.from_physical(
    lambda states: types.positions(x=states.positions.x(), y=states.positions.y())
)
```

## Planning Loop

A typical simulation loop:

```python
state = initial_state
nominal = initial_nominal_input

for step in range(max_steps):
    control = planner.step(
        temperature=50.0,
        nominal_input=nominal,
        initial_state=state,
    )

    state = model.step(inputs=control.optimal, state=state)
    nominal = control.nominal  # warm-start for next iteration
```

`control.optimal` is the weighted-average control sequence. `control.nominal` is the shifted and padded sequence used as the center of sampling in the next step.
