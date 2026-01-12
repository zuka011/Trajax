# trajax-visualizer

Interactive visualization tool for [trajax](https://gitlab.com/risk-metrics/trajax) trajectory planning simulations.

## Installation

```bash
pip install trajax-visualizer
```

Or with uv:

```bash
uv add trajax-visualizer
```

## Requirements

- **Python 3.13+**
- **Node.js 18+** (required at runtime for generating HTML visualizations)

## Usage

### Basic Usage

```python
from trajax_visualizer import visualizer, MpccSimulationResult

# Create a visualizer instance
mpcc_viz = visualizer.mpcc()

# After running your simulation, create a result object
result = MpccSimulationResult(
    reference=trajectory,
    states=states,
    optimal_trajectories=optimal_trajectories,
    nominal_trajectories=nominal_trajectories,
    contouring_errors=contouring_errors,
    lag_errors=lag_errors,
    wheelbase=wheelbase,
    max_contouring_error=max_contouring_error,
    max_lag_error=max_lag_error,
)

# Generate visualization
await mpcc_viz(result, key="my-simulation")
```

## Output

Visualizations are saved as:
- `<key>.json` - Raw simulation data
- `<key>.html` - Interactive HTML visualization with Plotly

## Development

The bundled CLI (`visualizer/trajax_visualizer/assets/cli.js`) is included in the package distribution. See `visualizer/core/README.md` for build instructions.
