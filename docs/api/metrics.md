# metrics

The metrics module provides evaluation metrics for trajectory planning algorithms.

## Available Metrics

### Collision Metric

Evaluates collision safety:

::: trajax.metrics.collision.CollisionMetric
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.metrics.collision.CollisionMetricResult
    options:
      show_root_heading: true
      heading_level: 3

### Comfort Metric

Evaluates ride comfort based on acceleration profiles:

::: trajax.metrics.comfort.ComfortMetric
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.metrics.comfort.ComfortMetricResult
    options:
      show_root_heading: true
      heading_level: 3

### Constraint Violation Metric

Evaluates boundary and constraint adherence:

::: trajax.metrics.constraint.ConstraintViolationMetric
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.metrics.constraint.ConstraintViolationMetricResult
    options:
      show_root_heading: true
      heading_level: 3

### MPCC Error Metric

Evaluates path-following error for MPCC controllers:

::: trajax.metrics.error.MpccErrorMetric
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.metrics.error.MpccErrorMetricResult
    options:
      show_root_heading: true
      heading_level: 3

### Task Completion Metric

Evaluates goal reaching and trajectory progress:

::: trajax.metrics.task.TaskCompletionMetric
    options:
      show_root_heading: true
      heading_level: 3

::: trajax.metrics.task.TaskCompletionMetricResult
    options:
      show_root_heading: true
      heading_level: 3

## Metric Registry

::: trajax.metrics.registry.MetricRegistry
    options:
      show_root_heading: true
      heading_level: 3
