---
hide:
  - navigation
---

# Examples

Each example below is a complete MPCC planning loop. Click the **Result** tab to see the interactive visualization, or expand **Full code** to see the entire script.

## Basic Path Following

Bicycle model following an S-curve with contouring, lag, and progress costs.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zuka011/Trajax/blob/main/notebooks/01_basic_path_following.ipynb){ .colab-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py:loop"
    ```

=== "Result"

    <iframe src="../../visualizations/mpcc-simulation/doc-basic-path-following.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/01_basic_path_following.py"
    ```


## Path Following with Boundaries

Same setup plus a fixed-width corridor (2.5 m each side). The boundary cost activates when the vehicle approaches the corridor edge.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zuka011/Trajax/blob/main/notebooks/02_path_following_with_boundaries.ipynb){ .colab-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py:loop"
    ```

=== "Result"

    <iframe src="../../visualizations/mpcc-simulation/doc-path-following-with-boundary.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/02_path_following_with_boundaries.py"
    ```


## Obstacle Avoidance

Three static obstacles along the path. Circle-to-circle distance computation with a hinge-loss collision cost and a fixed-width corridor boundary.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zuka011/Trajax/blob/main/notebooks/03_obstacle_avoidance.ipynb){ .colab-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py:loop"
    ```

=== "Result"

    <iframe src="../../visualizations/mpcc-simulation/doc-static-obstacles.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/03_obstacle_avoidance.py"
    ```


## Obstacle Avoidance with Uncertainty

Four moving obstacles with covariance propagation and a mean-variance risk metric. The planner accounts for prediction uncertainty when evaluating collision costs, producing more cautious trajectories.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zuka011/Trajax/blob/main/notebooks/04_obstacle_avoidance_with_uncertainty.ipynb){ .colab-badge }

=== "Setup"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:setup"
    ```

=== "Planning loop"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py:loop"
    ```

=== "Result"

    <iframe src="../../visualizations/mpcc-simulation/doc-dynamic-obstacles-uncertain.html" width="100%" height="800" frameborder="0"></iframe>

??? note "Full code"

    ```python
    --8<-- "docs/examples/04_obstacle_avoidance_with_uncertainty.py"
    ```
