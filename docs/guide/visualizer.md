# Visualizer

`faran-visualizer` is a separate package that generates interactive HTML visualizations from simulation results.

## Installation

```bash
pip install faran-visualizer
```

Requires **Node.js 18+** at runtime.

## MPCC Visualization

```python
from faran_visualizer import visualizer, MpccSimulationResult
from numtypes import array

result = MpccSimulationResult(
    reference=trajectory,
    states=augmented_states,
    contouring_errors=array(contouring_errors),
    lag_errors=array(lag_errors),
    time_step_size=0.1,
    wheelbase=2.5,
)

mpcc_viz = visualizer.mpcc()
await mpcc_viz(result, key="my-simulation")
```

`MpccSimulationResult` also accepts optional fields: `obstacles`, `obstacle_forecasts`, `boundary`, `controls`, `risks`, `optimal_trajectories`, `nominal_trajectories`, `vehicle_width`, `max_contouring_error`, `max_lag_error`, and `network`.

## Generic Visualization

For simulations that don't use the MPCC factory:

```python
from faran_visualizer import visualizer, Visualizable

result = Visualizable.SimulationResult.create(
    info=Visualizable.SimulationInfo(
        path_length=100.0, time_step=0.1, wheelbase=2.5,
    ),
    reference=Visualizable.ReferenceTrajectory(x=ref_x, y=ref_y),
    ego=Visualizable.Ego(
        x=ego_x, y=ego_y, heading=ego_heading, path_parameter=ego_progress,
    ),
)

sim_viz = visualizer.simulation()
await sim_viz(result, key="my-simulation")
```

## Custom Plots

Attach additional time-series plots with optional bounds and uncertainty bands:

```python
from faran_visualizer import Plot

speed_plot = Plot.Additional(
    id="speed",
    name="Vehicle Speed",
    series=[Plot.Series(label="Speed", values=speeds, color="blue")],
    y_axis_label="Speed (m/s)",
    upper_bound=Plot.Bound(values=15.0, label="Max Speed"),
)
```

## Output

Each visualization produces a `<key>.json` data file and a self-contained `<key>.html` file with interactive Plotly charts.
