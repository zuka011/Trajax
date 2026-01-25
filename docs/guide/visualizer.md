# Visualizer

trajax-visualizer generates interactive HTML visualizations for trajectory planning simulations.

## Installation

```bash
pip install trajax-visualizer
```

The visualizer requires **Node.js 18+** at runtime.

## MPCC Visualizer

```python
from trajax_visualizer import visualizer, MpccSimulationResult
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

### MpccSimulationResult Fields

```python
MpccSimulationResult(
    # Required
    reference=trajectory,
    states=augmented_states,
    contouring_errors=contouring_errors,
    lag_errors=lag_errors,
    time_step_size=0.1,
    wheelbase=2.5,

    # Optional
    max_contouring_error=0.5,
    max_lag_error=1.0,
    vehicle_width=1.2,
    optimal_trajectories=trajectories,
    nominal_trajectories=nominals,
    obstacles=obstacle_states,
    obstacle_forecasts=forecasts,
    controls=controls,
    risks=risks,
    network=road_network,
    boundary=corridor,
)
```

## Simulation Visualizer

For generic simulations:

```python
from trajax_visualizer import visualizer, Visualizable, Plot

result = Visualizable.SimulationResult.create(
    info=Visualizable.SimulationInfo(
        path_length=100.0,
        time_step=0.1,
        wheelbase=2.5,
    ),
    reference=Visualizable.ReferenceTrajectory(
        x=reference_x,
        y=reference_y,
    ),
    ego=Visualizable.Ego(
        x=ego_x,
        y=ego_y,
        heading=ego_heading,
        path_parameter=ego_progress,
    ),
)

sim_viz = visualizer.simulation()
await sim_viz(result, key="my-simulation")
```

## Custom Plots

```python
from trajax_visualizer import Plot

speed_plot = Plot.Additional(
    id="speed",
    name="Vehicle Speed",
    series=[
        Plot.Series(label="Speed", values=speeds, color="blue"),
        Plot.Series(label="Target", values=target_speeds, color="green"),
    ],
    y_axis_label="Speed (m/s)",
    upper_bound=Plot.Bound(values=15.0, label="Max Speed"),
)
```

### Uncertainty Bands

```python
risk_plot = Plot.Additional(
    id="risk",
    name="Collision Risk",
    series=[
        Plot.Series(label="Median", values=np.median(risks, axis=1)),
    ],
    bands=[
        Plot.Band(
            lower=np.percentile(risks, 25, axis=1),
            upper=np.percentile(risks, 75, axis=1),
            color="#457b9d",
            label="25-75%",
        ),
    ],
    y_axis_label="Risk",
    y_axis_scale="log",
)
```

## Output Files

| File | Description |
|------|-------------|
| `<key>.json` | Serialized simulation data |
| `<key>.html` | Interactive Plotly visualization |
