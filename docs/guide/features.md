# Feature Overview

Everything listed below is implemented, tested, and available in both the NumPy and JAX backends unless noted otherwise.

---

## :material-robot: Planning

<div class="grid cards" markdown>

-   **MPPI** · Model Predictive Path Integral

    ---

    Sampling-based trajectory optimizer. Configurable temperature, rollout count, horizon, and filtering.

    Three factory levels: [`mppi.base`](../api/mppi.md) (custom MPC), [`mppi.augmented`](../api/mppi.md) (physical + virtual states), [`mppi.mpcc`](../api/mppi.md) (path following).

    :material-check-all: NumPy  · :material-check-all: JAX

</div>

---

## :material-format-list-checks: MPC Formulations

<div class="grid cards" markdown>

-   **MPCC** · Model Predictive Contouring Control

    ---

    MPC formulation that decomposes tracking error into contouring (lateral) and lag (longitudinal) components, with a virtual path parameter driving progress.

    [:octicons-arrow-right-24: Cost design](costs.md) · [:octicons-arrow-right-24: Concepts](concepts.md#mpcc-model-predictive-contouring-control)

    :material-check-all: NumPy · :material-check-all: JAX

</div>

---

## :material-car-side: Dynamics Models

<div class="grid cards" markdown>

-   **Kinematic Bicycle**

    ---

    4-state model ($x, y, \theta, v$) with acceleration and steering inputs. Configurable wheelbase and input limits.

    [:octicons-arrow-right-24: API](../api/model.md)

-   **Unicycle**

    ---

    3-state model ($x, y, \theta$) with speed and angular velocity inputs.

    [:octicons-arrow-right-24: API](../api/model.md)

-   **Integrator**

    ---

    Generic $n$-dimensional single or double integrator. Used for virtual states in MPCC and for obstacle motion prediction.

    [:octicons-arrow-right-24: API](../api/model.md)

</div>

All models are available in both backends and support configurable state/input limits.

---

## :material-dice-multiple: Samplers

<div class="grid cards" markdown>

-   **Gaussian**

    ---

    Zero-mean Gaussian perturbations around a nominal control sequence. Per-dimension standard deviation.

    [:octicons-arrow-right-24: API](../api/sampler.md)

-   **Halton + Spline**

    ---

    Halton quasi-random sequences mapped through an inverse normal CDF and interpolated with cubic splines. Provides temporally smooth, low-discrepancy perturbations.

    [:octicons-arrow-right-24: API](../api/sampler.md)

</div>

---

## :material-function-variant: Cost Functions

All costs operate on batched state/input tensors of shape $(T, D, M)$ and return per-rollout costs.

<div class="grid cards" markdown>

-   **Tracking**

    ---

    - **Contouring** — penalizes lateral deviation from the reference path
    - **Lag** — penalizes longitudinal deviation from the reference point
    - **Progress** — rewards forward motion along the path

    [:octicons-arrow-right-24: Cost design](costs.md)

-   **Safety**

    ---

    - **Collision** — penalizes proximity to obstacles using signed distance computation
    - **Boundary** — penalizes states approaching corridor edges

    [:octicons-arrow-right-24: Obstacles](obstacles.md) · [:octicons-arrow-right-24: Boundaries](boundaries.md)

-   **Comfort**

    ---

    - **Control smoothing** — penalizes rate of change between consecutive inputs
    - **Control effort** — penalizes input magnitude

    [:octicons-arrow-right-24: Cost design](costs.md)

-   **Composition**

    ---

    `costs.combined(...)` sums any number of cost components. Custom cost functions can be any callable with the matching signature.

    [:octicons-arrow-right-24: Cost design](costs.md)

</div>

---

## :material-shield-alert: Collision Avoidance

<div class="grid cards" markdown>

-   **Distance Computation**

    ---

    - **Circle-to-circle** — fast distance between circle-based vehicle and obstacle representations
    - **SAT (Separating Axis Theorem)** — exact signed distance between convex polygons

    [:octicons-arrow-right-24: API](../api/obstacles.md)

-   **Obstacle Handling**

    ---

    - Static obstacles (fixed positions and headings)
    - Dynamic obstacles with constant-velocity prediction
    - Gaussian sampling for uncertainty propagation
    - Running history with ID-based tracking
    - Hungarian algorithm for obstacle ID assignment across frames

    [:octicons-arrow-right-24: Obstacle guide](obstacles.md)

</div>

---

## :material-chart-bell-curve: Risk Metrics

Risk-aware collision costs via the [riskit](https://gitlab.com/risk-metrics/riskit) library. The risk metric replaces the expected cost with a risk measure over sampled obstacle predictions.

| Metric | Description |
|---|---|
| **Expected value** | Mean cost over samples |
| **Mean-variance** | Mean + $\gamma \cdot$ variance |
| **VaR** | Value at Risk at confidence $\alpha$ |
| **CVaR** | Conditional Value at Risk at confidence $\alpha$ |
| **Entropic risk** | Exponential risk measure with parameter $\theta$ |

:material-check-all: NumPy · :material-check-all: JAX

---

## :material-road: Reference Trajectories & Boundaries

<div class="grid cards" markdown>

-   **Trajectories**

    ---

    - **Waypoints** — piecewise linear path through a sequence of 2D points
    - **Line** — straight path between two endpoints

    [:octicons-arrow-right-24: Trajectories](trajectories.md) · [:octicons-arrow-right-24: API](../api/trajectory.md)

-   **Boundaries**

    ---

    - **Fixed-width corridor** — symmetric or asymmetric constant width
    - **Piecewise fixed-width** — segment-varying corridor widths defined at arc-length breakpoints

    [:octicons-arrow-right-24: Boundaries](boundaries.md) · [:octicons-arrow-right-24: API](../api/boundary.md)

</div>

---

## :material-chart-line: Evaluation Metrics

Post-simulation evaluation for benchmarking and analysis.

| Metric | Measures |
|---|---|
| **Collision** | Minimum distances, collision detection per time step |
| **MPCC error** | Contouring and lag error over the trajectory |
| **Task completion** | Goal reached, completion time, stretch ratio, progress fraction |
| **Constraint violation** | Boundary and limit violations |
| **Comfort** | Jerk, lateral acceleration, smoothness |

[:octicons-arrow-right-24: Metrics guide](metrics.md) · [:octicons-arrow-right-24: API](../api/metrics.md)

---

## :material-swap-horizontal: Backend Support

Both backends expose identical factory functions. Switching requires changing one import line.

| | NumPy | JAX |
|---|:---:|:---:|
| All planning components | :material-check: | :material-check: |
| All cost functions | :material-check: | :material-check: |
| All evaluation metrics | :material-check: | :material-check: |
| GPU acceleration | — | :material-check: (Linux, CUDA) |
| JIT compilation | — | :material-check: |

[:octicons-arrow-right-24: Backend guide](backends.md)

---

## :material-map-marker-path: Roadmap

### Current Coverage

| Component | NumPy | JAX | Status |
|---|:---:|:---:|---|
| **Planning** | | | |
| MPPI (`mppi.base`) | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| MPPI Augmented (`mppi.augmented`) | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| MPCC (`mppi.mpcc`) | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| **Dynamics Models** | | | |
| Kinematic Bicycle | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Unicycle | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Integrator | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| **Samplers** | | | |
| Gaussian | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Halton + Spline | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| **Cost Functions** | | | |
| Contouring / Lag / Progress | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Collision (hinge-loss) | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Boundary | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Control Smoothing / Effort | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| **Collision Avoidance** | | | |
| Circle-to-Circle Distance | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| SAT Distance | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Static Obstacles | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Dynamic Obstacles | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Hungarian ID Assignment | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| **Risk Metrics** | | | |
| Expected Value | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Mean-Variance | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| VaR | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| CVaR | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Entropic Risk | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| **Trajectories & Boundaries** | | | |
| Waypoints | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Line | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Fixed-Width Boundary | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Piecewise Fixed-Width Boundary | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| **Evaluation Metrics** | | | |
| Collision | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| MPCC Error | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Task Completion | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Constraint Violation | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| Comfort | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |
| **Visualization** | | | |
| Interactive HTML (faran-visualizer) | :material-check-all:{ .green } | :material-check-all:{ .green } | :material-tag: Stable |

### Planned

| Feature | NumPy | JAX | Status |
|---|:---:|:---:|---|
| Additional planning algorithms (iLQR) | :material-progress-clock:{ .amber } | :material-progress-clock:{ .amber } | :material-hammer-wrench: Planned |
| Spline-based reference trajectories | :material-progress-clock:{ .amber } | :material-progress-clock:{ .amber } | :material-hammer-wrench: Planned |
| Multi-agent / human environments | :material-check:{ .green } | :material-progress-clock:{ .amber } | :material-wrench: Partial |
