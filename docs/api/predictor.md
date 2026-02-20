# predictor

The predictor module provides motion prediction for dynamic obstacles, with optional covariance propagation built into the obstacle models.

## Overview

Predictors estimate future obstacle states based on motion models. When using Kalman filter-based estimators, covariance propagation is automatically performed by the obstacle model.

## Usage

=== "NumPy"

    ```python
    from faran import predictor, model

    # Create a motion predictor with EKF estimator (includes covariance propagation)
    motion_predictor = predictor.numpy.curvilinear(
        horizon=10,
        model=model.numpy.bicycle.obstacle(time_step_size=0.1, wheelbase=2.8),
        estimator=model.numpy.bicycle.estimator.ekf(
            time_step_size=0.1,
            wheelbase=2.8,
            process_noise_covariance=0.01,
            observation_noise_covariance=0.01,
        ),
        prediction=...,  # Your prediction creator
    )
    ```

=== "JAX"

    ```python
    from faran import predictor, model

    motion_predictor = predictor.jax.curvilinear(
        horizon=10,
        model=model.jax.bicycle.obstacle(time_step_size=0.1, wheelbase=2.8),
        estimator=model.jax.bicycle.estimator.ekf(
            time_step_size=0.1,
            wheelbase=2.8,
            process_noise_covariance=0.01,
            observation_noise_covariance=0.01,
        ),
        prediction=...,  # Your prediction creator
    )
    ```

## Motion Predictor Protocol

::: faran.types.ObstacleMotionPredictor
    options:
      show_root_heading: true
      heading_level: 3

## Input Assumption Provider Protocol

::: faran.types.InputAssumptionProvider
    options:
      show_root_heading: true
      heading_level: 3

## Predicting Obstacle State Provider

Combines prediction with state observation:

::: faran.obstacles.PredictingObstacleStateProvider
    options:
      show_root_heading: true
      heading_level: 3
