---
hide:
  - navigation
---

# Examples

Complete [MPCC](concepts.md#mpcc-model-predictive-contouring-control) planning scenarios, each building on the previous one. Click the **Result** tab to see the interactive visualization, or expand **Full code** to see the entire script.

Each example can also be run as a Jupyter notebook via the Binder badge.

## Basic Path Following

The simplest case: a [bicycle model](models.md#kinematic-bicycle-model) following an S-curve using contouring, lag, and progress costs. This example introduces the core planning loop and demonstrates how [MPCC](concepts.md#mpcc-model-predictive-contouring-control) tracks a reference trajectory.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zuka011/faran/main?filepath=notebooks/01_basic_path_following.ipynb){ .binder-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py:loop"
    ```

=== "Visualization"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py:visualize"
    ```

=== "Result"

    <iframe src="../../visualizations/mpcc-simulation/doc-basic-path-following.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py"
    ```


## Path Following with Boundaries

Building on the basic example, this adds a fixed-width corridor (2.5 m each side). The [boundary cost](costs.md#boundary) activates when the vehicle approaches the corridor edge, steering it away from violations. Notice how the vehicle stays inside the corridor even on tight turns.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zuka011/faran/main?filepath=notebooks/02_path_following_with_boundaries.ipynb){ .binder-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py:loop"
    ```

=== "Visualization"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py:visualize"
    ```

=== "Result"

    <iframe src="../../visualizations/mpcc-simulation/doc-path-following-with-boundary.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py"
    ```


## Obstacle Avoidance

This example adds three static obstacles along the path. It demonstrates the full avoidance pipeline: [circle-based distance computation](obstacles.md#circle-to-circle), a hinge-loss [collision cost](costs.md#collision), and a fixed-width corridor boundary. The planner swerves around obstacles while staying inside the corridor.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zuka011/faran/main?filepath=notebooks/03_obstacle_avoidance.ipynb){ .binder-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py:loop"
    ```

=== "Visualization"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py:visualize"
    ```

=== "Result"

    <iframe src="../../visualizations/mpcc-simulation/doc-static-obstacles.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py"
    ```


## Obstacle Avoidance with Uncertainty

The most advanced example: four moving obstacles with covariance propagation and a mean-variance risk metric. The planner uses [state estimation](../api/predictor.md) to track obstacles, propagates uncertainty through motion predictions, and evaluates collision risk using sampled obstacle poses. The result is a more cautious trajectory that accounts for prediction uncertainty — the vehicle gives wider berth to uncertain obstacles.

[![Launch Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/zuka011/faran/main?filepath=notebooks/04_obstacle_avoidance_with_uncertainty.ipynb){ .binder-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:loop"
    ```

=== "Visualization"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:visualize"
    ```

=== "Result"

    <iframe src="../../visualizations/mpcc-simulation/doc-dynamic-obstacles-uncertain.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py"
    ```
