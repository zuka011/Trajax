# collectors

The collectors module provides data collection utilities for simulation and analysis.

## Overview

Collectors capture data during MPPI simulation runs for later analysis and visualization.

## Usage

```python
from faran import collectors, access

# Create a registry for data collection
registry = collectors.CollectorRegistry()

# Access collected data after simulation
data = access(registry)
```

## CollectorRegistry

::: faran.collectors.registry.CollectorRegistry
    options:
      show_root_heading: true
      heading_level: 3

## Data Access

::: faran.collectors.access
    options:
      show_root_heading: true
      heading_level: 3

## Warning Types

::: faran.collectors.NoCollectedDataWarning
    options:
      show_root_heading: true
      heading_level: 3
