# collectors

The collectors module provides data collection utilities for simulation and analysis.

## Overview

Collectors capture data during MPPI simulation runs for later analysis and visualization.

## Usage

```python
from trajax import collectors, access

# Create a registry for data collection
registry = collectors.CollectorRegistry()

# Access collected data after simulation
data = access(registry)
```

## CollectorRegistry

::: trajax.collectors.registry.CollectorRegistry
    options:
      show_root_heading: true
      heading_level: 3

## Data Access

::: trajax.collectors.access
    options:
      show_root_heading: true
      heading_level: 3

## Warning Types

::: trajax.collectors.NoCollectedDataWarning
    options:
      show_root_heading: true
      heading_level: 3
