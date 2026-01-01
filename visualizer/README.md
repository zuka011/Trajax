# Simulation Visualizer

TypeScript CLI for generating interactive HTML visualizations of mobile robot simulations.

## Setup

```bash
pnpm install && pnpm build
```

## Usage

```bash
# Generate visualization
node dist/cli/index.js generate <input.json> -o output.html
```

## Input Format

See `src/core/types.ts` for the full schema. Minimum required:

```json
{
  "reference": { "x": [], "y": [] },
  "positions_x": [],
  "positions_y": [],
  "headings": [],
  "path_parameters": [],
  "path_length": 0
}
```
