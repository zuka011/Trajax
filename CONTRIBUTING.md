# Contributing to faran

Thank you for your interest in contributing to faran! This guide will help you get started.

## Development Setup

### Prerequisites

- Python ≥ 3.13
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [just](https://github.com/casey/just) (optional, for task running)

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://gitlab.com/risk-metrics/faran.git
   cd faran
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv sync
   ```

   This installs all dependency groups: `dev`, `test`, `benchmark`, and `doc`.

3. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Code Style

### General Principles

1. **Decomposition over comments** — Use well-named functions and classes to explain your code rather than comments.

2. **Descriptive names** — Use full words in identifiers. You may omit words when context is clear, but never abbreviate.
   ```python
   # Good
   control_input_sequence = ...
   rollout_count = ...
   
   # Bad
   ctrl_inp_seq = ...
   n_rollouts = ...
   ```

3. **Immutability** — Prefer immutable data structures. Use `frozen=True` in dataclasses.
   ```python
   @dataclass(frozen=True)
   class BicycleState:
       x: float
       y: float
       heading: float
       speed: float
   ```

4. **Functional style** — Prefer declarative/functional patterns over imperative loops.

5. **Static typing** — All code must be fully typed. We use:
   - `pyright` for type checking
   - `beartype` for runtime validation
   - `jaxtyping` for array shape annotations in JAX code

### Formatting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check and auto-fix
uv run ruff check --fix

# Format
uv run ruff format
```

### Type Checking

```bash
uv run pyright
```

### Pre-commit

All checks run automatically on commit via pre-commit hooks. To run manually:

```bash
uv run pre-commit run --all-files
```

## Testing

### Test Naming Convention

Tests follow this naming pattern:

```
test_that_<functionality>[_when_<condition>]
```

Examples:
- `test_that_mppi_favors_samples_with_lower_costs`
- `test_that_tracking_cost_does_not_depend_on_coordinate_system`
- `test_that_query_returns_first_waypoint_when_path_parameter_is_zero`

### Test Structure

Tests use the **Arrange-Act-Assert** pattern and are organized as classes:

```python
class test_that_mppi_favors_samples_with_lower_costs:
    @staticmethod
    def cases(create_mppi, data, costs) -> Sequence[tuple]:
        return [
            (
                # Arrange: Set up test data
                mppi := create_mppi.base(...),
                temperature := 0.1,
                nominal_input,
                initial_state,
                expected := ...,
                tolerance := 1e-3,
            ),
        ]

    @mark.parametrize(
        ["mppi", "temperature", "nominal_input", "initial_state", "expected", "tolerance"],
        [
            *cases(create_mppi=create_mppi.numpy, data=data.numpy, costs=costs.numpy),
            *cases(create_mppi=create_mppi.jax, data=data.jax, costs=costs.jax),
        ],
    )
    def test(self, mppi, temperature, nominal_input, initial_state, expected, tolerance):
        # Act
        result = mppi.step(
            temperature=temperature,
            nominal_input=nominal_input,
            initial_state=initial_state,
        )
        
        # Assert
        assert np.allclose(result.optimal.array, expected.array, atol=tolerance)
```

### Dual-Backend Testing

All functionality must work identically on both NumPy and JAX backends. Use parameterized tests that run against both:

```python
@mark.parametrize(
    ["trajectory", "expected"],
    [
        *cases(trajectory=trajectory.numpy),
        *cases(trajectory=trajectory.jax),
    ],
)
def test(self, trajectory, expected):
    ...
```

### Test DSL

We use a custom test DSL in `tests/dsl/` to create test data. This keeps tests readable:

```python
from tests.dsl import mppi as data, costs, stubs

# Create test data using the DSL
states = data.state_batch(array(...))
inputs = data.control_input_batch(array(...))
cost = costs.energy()
```

### No Mocks

**Never use mocks.** Use stubs or real implementations:

```python
from tests.dsl import stubs

# Stub that returns predetermined values
model = stubs.DynamicalModel.returns(
    rollouts=expected_rollouts,
    when_control_inputs_are=inputs,
    and_initial_state_is=initial_state,
)
```

### Running Tests

```bash
# Run unit tests
uv run pytest

# Run with coverage
uv run pytest --cov=faran --cov-report=term-missing

# Run specific test
uv run pytest tests/test_mppi.py -k "test_that_mppi_favors"

# Run integration tests
uv run pytest -m integration

# Run benchmarks
uv run pytest -m benchmark --benchmark-json=benchmark.json
```

### Test-Driven Development

We follow TDD principles:

1. **Write the test first** — Define expected behavior before implementation
2. **Watch it fail** — Verify the test fails for the right reason
3. **Implement minimally** — Write just enough code to pass
4. **Refactor** — Clean up while keeping tests green

## Documentation

### Building Docs

```bash
uv run mkdocs serve
```

Visit `http://127.0.0.1:8000/faran/` to preview.

### Docstrings

Use Google-style docstrings:

```python
def simulate(
    self,
    *,
    inputs: ControlInputBatch,
    initial_state: State,
) -> StateBatch:
    """Simulate the dynamical model forward in time.

    Args:
        inputs: Control inputs for each rollout.
        initial_state: Starting state for all rollouts.

    Returns:
        State trajectories for each rollout.

    Example:
        >>> model = bicycle.dynamical(time_step_size=0.1, wheelbase=2.5)
        >>> states = model.simulate(inputs=samples, initial_state=start)
    """
```

## Pull Request Process

1. **Create a branch** from `main` with a descriptive name
2. **Make your changes** following the style guidelines
3. **Add tests** for new functionality
4. **Run the full check suite**:
   ```bash
   just check
   # Or manually:
   uv run ruff check --fix && uv run ruff format && uv run pyright && uv run pytest
   ```
5. **Update documentation** if needed
6. **Submit a merge request** with a clear description

## Architecture Overview

```
faran/
├── __init__.py          # Public API exports
├── numpy.py             # NumPy backend namespace
├── jax.py               # JAX backend namespace
├── costs/               # Cost function implementations
├── models/              # Dynamical models (bicycle, unicycle, etc.)
├── mppi/                # MPPI planner (basic + accelerated)
├── obstacles/           # Obstacle handling and collision
├── samplers/            # Control input samplers
├── trajectories/        # Reference trajectory representations
└── types/               # Type definitions and protocols
```

### Backend Parity

Both backends must provide identical APIs. Changes to one backend should be mirrored in the other.

## Questions?

Open an issue on GitLab or reach out to the maintainers.
