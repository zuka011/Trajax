# metrics

The metrics module provides evaluation metrics for trajectory planning algorithms.

## Available Metrics

### Collision Metric

Evaluates collision safety:

::: faran.metrics.collision.CollisionMetric
    options:
      show_root_heading: true
      heading_level: 3

::: faran.metrics.collision.CollisionMetricResult
    options:
      show_root_heading: true
      heading_level: 3

### Comfort Metric

Evaluates ride comfort based on acceleration profiles:

::: faran.metrics.comfort.ComfortMetric
    options:
      show_root_heading: true
      heading_level: 3

::: faran.metrics.comfort.ComfortMetricResult
    options:
      show_root_heading: true
      heading_level: 3

### Constraint Violation Metric

Evaluates boundary and constraint adherence:

::: faran.metrics.constraint.ConstraintViolationMetric
    options:
      show_root_heading: true
      heading_level: 3

::: faran.metrics.constraint.ConstraintViolationMetricResult
    options:
      show_root_heading: true
      heading_level: 3

### MPCC Error Metric

Evaluates path-following error for MPCC controllers:

::: faran.metrics.error.MpccErrorMetric
    options:
      show_root_heading: true
      heading_level: 3

::: faran.metrics.error.MpccErrorMetricResult
    options:
      show_root_heading: true
      heading_level: 3

### Task Completion Metric

Evaluates goal reaching and trajectory progress:

::: faran.metrics.task.TaskCompletionMetric
    options:
      show_root_heading: true
      heading_level: 3

::: faran.metrics.task.TaskCompletionMetricResult
    options:
      show_root_heading: true
      heading_level: 3

## Metric Registry

::: faran.metrics.registry.MetricRegistry
    options:
      show_root_heading: true
      heading_level: 3
