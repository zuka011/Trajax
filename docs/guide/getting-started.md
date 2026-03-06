# Getting Started

By the end of this page you'll have a working MPPI planner tracking a reference path. This will take about 10 minutes.

## Installation

```bash
pip install faran          # NumPy + JAX (CPU)
pip install faran[cuda]    # JAX with GPU support (Linux)
```

Requires Python 3.13+. 

The visualizer is a separate optional package:

```bash
pip install faran-visualizer
```

Verify the installation:

```bash
python -c "import faran; print(faran.__version__)"
```

## Your First Planner

We'll build a planner that follows a curved reference path using a [kinematic bicycle model](models.md) — the same setup shown in the README.

The fastest way to get there is [`mppi.mpcc()`](../api/mppi.md). It assembles an [MPPI](mppi.md) planner with contouring, lag, and progress costs for path following using the [MPCC formulation](concepts.md#mpcc-model-predictive-contouring-control).

### Setup

```python
--8<-- "docs/examples/01_basic_path_following.py:setup"
```

This creates four objects:

| Object            | What it is                                                      |
|-------------------|-----------------------------------------------------------------|
| `planner`         | The MPPI planner — call `.step()` to get controls               |
| `augmented_model` | Combined physical + virtual dynamics model                      |
| `contouring_cost` | Contouring cost component, for [evaluation metrics](metrics.md) |
| `lag_cost`        | Lag cost component, for [evaluation metrics](metrics.md)        |

### Simulation Loop

Run the planner in a loop. Each call to `planner.step()` samples control sequences, evaluates their costs, and returns:

- `control.optimal` — the best control sequence for this step
- `control.nominal` — the shifted sequence used as the sampling center
  for the next step (warm-starting)

```python
--8<-- "docs/examples/01_basic_path_following.py:loop"
```

### Result

After 100 steps the vehicle will have tracked the reference path. If you have [`faran-visualizer`](visualizer.md) installed, you can generate an interactive visualization:

```python
--8<-- "docs/examples/01_basic_path_following.py:visualize"
```

This produces a standalone HTML file you can open in any browser:

<iframe src="../../visualizations/mpcc-simulation/doc-basic-path-following.html"
        width="100%" height="500px" frameborder="0"></iframe>

??? note "Full example"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py"
    ```

## How MPCC Works

MPCC augments the vehicle state with a virtual path parameter $\phi$ that tracks progress along a [reference trajectory](trajectories.md):

| Component | State                | Controls         |
|-----------|----------------------|------------------|
| Physical  | $[x, y, \theta, v]$ | $[a, \delta]$   |
| Virtual   | $[\phi]$            | $[\dot{\phi}]$ |

Three [costs](costs.md) drive path following:

- **Contouring** — penalizes lateral deviation from the reference
- **Lag** — penalizes longitudinal offset between $\phi$ and the vehicle's projection
- **Progress** — rewards forward motion along the path ($\dot\phi > 0$)

The balance between these three costs determines tracking behavior. High contouring weight keeps the vehicle close to the path; high progress weight makes it drive faster.

!!! tip "Need more control?"

    For manual assembly with custom models, additional costs, or mixed samplers, see [Core Concepts](concepts.md).

## Next Steps

- **[Core Concepts](concepts.md)** — Understand how MPPI and MPCC work under the hood, and how to assemble a planner manually.
- **[MPPI Planning](mppi.md)** — Tune temperature, filtering, and seeding for better planner performance.
- **[Cost Function Design](costs.md)** — Add safety, comfort, or custom objectives beyond basic path tracking.
- **[Obstacle Handling](obstacles.md)** — Add collision avoidance with distance functions and boundary constraints.
- **[Examples](examples.md)** — See complete scenarios with interactive visualizations.
