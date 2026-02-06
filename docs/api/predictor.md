# predictor

The predictor module provides motion prediction and covariance propagation for dynamic obstacles.

## Overview

Predictors estimate future obstacle states based on motion models and covariance propagation.

## Usage

=== "NumPy"

    ```python
    from trajax import predictor, propagator

    # Create a motion predictor
    motion_predictor = predictor.numpy.constant_velocity(time_step_size=0.1)

    # Create a covariance propagator
    cov_propagator = propagator.numpy.constant_velocity(
        time_step_size=0.1,
        process_noise=0.01,
    )
    ```

=== "JAX"

    ```python
    from trajax import predictor, propagator

    motion_predictor = predictor.jax.constant_velocity(time_step_size=0.1)
    cov_propagator = propagator.jax.constant_velocity(
        time_step_size=0.1,
        process_noise=0.01,
    )
    ```

## Motion Predictor Protocol

::: trajax.types.ObstacleMotionPredictor
    options:
      show_root_heading: true
      heading_level: 3

## Covariance Propagator Protocol

::: trajax.types.CovariancePropagator
    options:
      show_root_heading: true
      heading_level: 3

## Predicting Obstacle State Provider

Combines prediction with state observation:

::: trajax.obstacles.PredictingObstacleStateProvider
    options:
      show_root_heading: true
      heading_level: 3
