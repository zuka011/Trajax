# visualizer

The trajax-visualizer package provides interactive HTML visualizations for MPCC simulation results.

!!! note "Separate Package"
    The visualizer is distributed as a separate package: `trajax-visualizer`

## Installation

```bash
pip install trajax-visualizer
```

## Factory Functions

```python
from trajax_visualizer import visualizer

# Create an MPCC visualizer
mpcc_viz = visualizer.mpcc(
    output="simulation",
    output_directory="./output",
)
```

## MpccSimulationResult

::: trajax_visualizer.api.mpcc.MpccSimulationResult
    options:
      show_root_heading: true
      heading_level: 3

## MpccVisualizer

::: trajax_visualizer.api.mpcc.MpccVisualizer
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - __call__

## Plot Types

Helper classes for adding custom plots:

### Plot.Series

::: trajax_visualizer.api.simulation.Plot.Series
    options:
      show_root_heading: true
      heading_level: 3

### Plot.Additional

::: trajax_visualizer.api.simulation.Plot.Additional
    options:
      show_root_heading: true
      heading_level: 3

### Plot.Bound

::: trajax_visualizer.api.simulation.Plot.Bound
    options:
      show_root_heading: true
      heading_level: 3

### Plot.Band

::: trajax_visualizer.api.simulation.Plot.Band
    options:
      show_root_heading: true
      heading_level: 3

## Road Network

For visualizing road networks:

### Road.Lane

::: trajax_visualizer.api.simulation.Road.Lane
    options:
      show_root_heading: true
      heading_level: 3

### Road.Network

::: trajax_visualizer.api.simulation.Road.Network
    options:
      show_root_heading: true
      heading_level: 3
