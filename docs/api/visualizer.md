# visualizer

The faran-visualizer package provides interactive HTML visualizations for MPCC simulation results.

!!! note "Separate Package"
    The visualizer is distributed as a separate package: `faran-visualizer`

## Installation

```bash
pip install faran-visualizer
```

## Factory Functions

```python
from faran_visualizer import visualizer

# Create an MPCC visualizer
mpcc_viz = visualizer.mpcc(
    output="simulation",
    output_directory="./output",
)
```

## MpccSimulationResult

::: faran_visualizer.api.mpcc.MpccSimulationResult
    options:
      show_root_heading: true
      heading_level: 3

## MpccVisualizer

::: faran_visualizer.api.mpcc.MpccVisualizer
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - create
        - __call__

## Plot Types

Helper classes for adding custom plots:

### Plot.Series

::: faran_visualizer.api.simulation.Plot.Series
    options:
      show_root_heading: true
      heading_level: 3

### Plot.Additional

::: faran_visualizer.api.simulation.Plot.Additional
    options:
      show_root_heading: true
      heading_level: 3

### Plot.Bound

::: faran_visualizer.api.simulation.Plot.Bound
    options:
      show_root_heading: true
      heading_level: 3

### Plot.Band

::: faran_visualizer.api.simulation.Plot.Band
    options:
      show_root_heading: true
      heading_level: 3

## Road Network

For visualizing road networks:

### Road.Lane

::: faran_visualizer.api.simulation.Road.Lane
    options:
      show_root_heading: true
      heading_level: 3

### Road.Network

::: faran_visualizer.api.simulation.Road.Network
    options:
      show_root_heading: true
      heading_level: 3
