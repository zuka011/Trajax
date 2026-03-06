# Feature Overview

Everything listed below is implemented, tested, and available in both the NumPy and JAX backends unless noted otherwise.

---

## :material-robot: Planning

<div class="grid cards" markdown>

-   **MPPI** · Model Predictive Path Integral

    ---

    Sampling-based trajectory optimizer. Configurable temperature, rollout count, horizon, and filtering.

    Three factory levels: [`mppi.base`](../api/mppi.md) (custom MPC), [`mppi.augmented`](../api/mppi.md) (physical + virtual states), [`mppi.mpcc`](../api/mppi.md) (MPCC path following).

    [:octicons-arrow-right-24: MPPI guide](mppi.md)

</div>

---

## :material-format-list-checks: MPC Formulations

<div class="grid cards" markdown>

-   **MPCC** · Model Predictive Contouring Control

    ---

    MPC formulation that decomposes tracking error into contouring (lateral) and lag (longitudinal) components, with a virtual path parameter driving progress.

    [:octicons-arrow-right-24: Cost design](costs.md) ·
    [:octicons-arrow-right-24: Concepts](concepts.md#mpcc-model-predictive-contouring-control)

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

---

## :material-dice-multiple: Samplers

<div class="grid cards" markdown>

-   **Gaussian**

    ---

    Zero-mean Gaussian perturbations around a nominal control sequence. Per-dimension standard deviation.

    [:octicons-arrow-right-24: API](../api/sampler.md)

-   **Halton + Spline**

    ---

    Halton quasi-random sequences mapped through an inverse normal CDF and interpolated with cubic splines. Temporally smooth, low-discrepancy perturbations.

    [:octicons-arrow-right-24: API](../api/sampler.md)

</div>

---

## :material-function-variant: Cost Functions

<div class="grid cards" markdown>

-   **Tracking**

    ---

    - **Contouring** — lateral deviation from the reference path
    - **Lag** — longitudinal deviation from the reference point
    - **Progress** — rewards forward motion along the path

    [:octicons-arrow-right-24: Cost design](costs.md)

-   **Safety**

    ---

    - **Collision** — proximity to obstacles via signed distance
    - **Boundary** — states approaching corridor edges

    [:octicons-arrow-right-24: Obstacles](obstacles.md) ·
    [:octicons-arrow-right-24: Boundaries](boundaries.md)

-   **Comfort**

    ---

    - **Control smoothing** — rate of change between consecutive inputs
    - **Control effort** — input magnitude

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

    - **Circle-to-circle** — fast distance between circular representations
    - **SAT** — exact signed distance between convex polygons

    [:octicons-arrow-right-24: API](../api/obstacles.md)

-   **Obstacle Handling**

    ---

    - Hungarian algorithm for obstacle ID assignment across frames
    - Running history with ID-based tracking
    - State prediction with model assumptions (e.g. constant velocity)
    - Gaussian sampling of predicted obstacle states for risk-aware planning

    [:octicons-arrow-right-24: Obstacle guide](obstacles.md)

</div>

---

## :material-chart-bell-curve: Risk Metrics

Risk-aware collision costs via the [riskit](https://gitlab.com/risk-metrics/riskit) library. The risk metric defines how a stochastic cost distribution is aggregated into a scalar cost for optimization. 

| Metric             | Description                                      |
|--------------------|--------------------------------------------------|
| **Expected value** | Mean cost over samples                           |
| **Mean-variance**  | Mean + $\gamma \cdot$ variance                   |
| **VaR**            | Value at Risk at confidence $\alpha$             |
| **CVaR**           | Conditional Value at Risk at confidence $\alpha$ |
| **Entropic risk**  | Exponential risk measure with parameter $\theta$ |

---

## :material-road: Reference Trajectories & Boundaries

<div class="grid cards" markdown>

-   **Trajectories**

    ---

    - **Waypoints** — a B-spline path defined by a sequence of waypoints
    - **Line** — straight path between two endpoints

    [:octicons-arrow-right-24: API](../api/trajectory.md)

-   **Boundaries**

    ---

    - **Fixed-width corridor** — symmetric or asymmetric constant width
    - **Piecewise fixed-width** — segment-varying widths at arc-length breakpoints

    [:octicons-arrow-right-24: API](../api/boundary.md)

</div>

---

## :material-chart-line: Evaluation Metrics

Post-simulation evaluation for benchmarking and analysis.

| Metric                   | Measures                                                        |
|--------------------------|-----------------------------------------------------------------|
| **Collision**            | Minimum distances, collision detection per time step            |
| **MPCC error**           | Contouring and lag error over the trajectory                    |
| **Task completion**      | Goal reached, completion time, stretch ratio, progress fraction |
| **Constraint violation** | Boundary and limit violations                                   |
| **Comfort**              | Jerk, lateral acceleration, smoothness                          |

[:octicons-arrow-right-24: Metrics guide](metrics.md) ·
[:octicons-arrow-right-24: API](../api/metrics.md)

---

## :material-map-marker-path: Roadmap

| Feature                               | Status  |
|---------------------------------------|---------|
| Additional planning algorithms (iLQR) | Planned |
| Waypoint formulation of MPC           | Planned |
| Multi-agent / human environments      | Partial |
