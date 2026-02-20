# obstacles

The obstacles module provides obstacle representation and handling for risk-aware planning.

## Overview

This module handles:

- Obstacle state representation and tracking
- Dynamic obstacle motion prediction
- Obstacle state sampling for uncertainty handling

## Usage

=== "NumPy"

    ```python
    from faran import obstacles, types

    # Create obstacle states from observed data
    states = types.numpy.obstacle_states(
        x=...,  # x-positions [T, K]
        y=...,  # y-positions [T, K]
        heading=...,  # headings [T, K]
        covariance=...,  # uncertainty [T, 3, 3, K]
    )

    # Create a Gaussian sampler for obstacle predictions
    sampler = obstacles.numpy.sampler.gaussian(seed=42)
    samples = sampler(states, count=N)
    ```

=== "JAX"

    ```python
    from faran import obstacles, types

    states = types.jax.obstacle_states(...)
    sampler = obstacles.jax.sampler.gaussian(seed=42)
    ```

## Obstacle Simulators

For testing and simulation purposes:

### Static Obstacle Simulator

::: faran.obstacles.static.basic.NumPyStaticObstacleSimulator
    options:
      show_root_heading: true
      heading_level: 3

## State Provider

::: faran.types.ObstacleStateProvider
    options:
      show_root_heading: true
      heading_level: 3

## ID Assignment

### Hungarian Algorithm

Matches detected obstacles to tracked IDs across frames using the Hungarian algorithm on position distances. Detections beyond a configurable cutoff distance are assigned new IDs.

::: faran.obstacles.assignment.basic.NumPyHungarianObstacleIdAssignment
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

::: faran.obstacles.assignment.accelerated.JaxHungarianObstacleIdAssignment
    options:
      show_root_heading: true
      heading_level: 4
      members:
        - create

## ID Assignment

::: faran.types.ObstacleIdAssignment
    options:
      show_root_heading: true
      heading_level: 3
