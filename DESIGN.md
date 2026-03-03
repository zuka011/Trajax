# Design Guide

This document describes the design principles, code style, and testing philosophy used in Faran. It's intended as a reference for contributors who want to understand the "why" behind our conventions. It's important to note that there could exist cases where deviating from these guidelines is justified, so use your judgement and feel free to discuss any such cases in an MR or issue.

This is **not** a checklist of rules to follow, so you don't need to read the whole thing.

## Code Style

### General Principles

1. **Decomposition over comments** — Use well-named functions and classes to explain your code rather than comments. AI assistants (e.g. GitHub Copilot) tend to generate verbosely commented and badly structured code, so be sure to pay extra attention to this if you do use AI assistance.

2. **Descriptive names** — Use full words in identifiers. You may omit words when context is clear, but never abbreviate.
   ```python
    # Good
    control_input_sequence = ...
    rollout_count = ...
    
    # Bad
    ctrl_inp_seq = ...
    n_rollouts = ...
   ```
    The minor convenience of shorter names is not worth the cognitive overhead of deciphering abbreviations, especially for those unfamiliar with the codebase.

3. **Immutability** — Prefer immutable data structures, e.g. frozen dataclasses or namedtuples:
   ```python
    # Good
    @dataclass(frozen=True)
    class BicycleState:
        x: float
        y: float
        heading: float
        speed: float

    # Also good
    class BicycleState(NamedTuple):
         x: float
         y: float
         heading: float
         speed: float

    # Bad, and verbose
    class BicycleState:
        def __init__(self, x: float, y: float, heading: float, speed: float):
            self.x = x
            self.y = y
            self.heading = heading
            self.speed = speed
   ```
    State makes components harder to test and reason about. Also, most Faran components are designed to be stateless anyways (including planners).

4. **Functional style** — Prefer declarative/functional patterns over imperative loops. The reasoning is similar to the previous point about immutability. Additionally, such a style makes porting implementations of algorithms to some backends easier (e.g. JAX).

5. **Static typing** — All code must be fully typed. We use:
   - `pyright` for type checking
   - `beartype` for runtime validation
   - `jaxtyping` for array shape annotations
   
   In reasonable cases, you may omit type annotations (e.g. in tests) or use `Any`, `# type: ignore` when the static typing is doing more harm than good.

## Testing Philosophy

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

Tests typically use the **Arrange-Act-Assert** pattern:

```python
def test_that_mppi_favors_samples_with_lower_costs():
    # Arrange: Set up test data
    mppi = create_mppi(...)
    temperature = 0.1
    nominal_input = ...
    initial_state = ...
    expected = ...
    tolerance = 1e-3

    # Act: Call the function under test
    result = mppi.step(
        temperature=temperature,
        nominal_input=nominal_input,
        initial_state=initial_state,
    )
    
    # Assert: Check the results
    assert np.allclose(result.optimal.array, expected.array, atol=tolerance)
```

### Parametrized Tests & Testing Multiple Backends

Sometimes we use parameterized tests to run the same logic against multiple components. If the parameterization logic is complicated, or the test case generation itself needs to be parameterized (e.g. to run against components with different backends), we organize the test as a class instead:

```python
class test_that_mppi_favors_samples_with_lower_costs:
    @staticmethod
    def cases(create_mppi, data, costs) -> Sequence[tuple]:
        return [
            (
                # Arrange: Set up test data agnostic to backend
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
    def test(
        self, 
        mppi: Mppi, 
        temperature: float, 
        nominal_input: ControlInputSequence, 
        initial_state: State, 
        expected: ControlInputSequence, 
        tolerance: float
    ) -> None:
        # Act
        result = mppi.step(
            temperature=temperature,
            nominal_input=nominal_input,
            initial_state=initial_state,
        )
        
        # Assert
        assert np.allclose(result.optimal.array, expected.array, atol=tolerance)
```

All functionality must work identically on all backends. Using parameterized tests like this also ensures components implemented using different backends have similar behavior (ignoring small numerical differences) and a common API.

```python
class test_that_...:
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

### Subtests

If you have a complex setup, but want to check multiple different behaviors/properties of a component, you can use subtests:

```python
from pytest import SubTests

def test_that_prediction_error_covariance_is_positive_definite(subtests: SubTests) -> None:
    # Arrange
    model = create_model(...)
    inputs = ...
    initial_state = ...
    
    # Act
    result = model.predict(inputs=inputs, initial_state=initial_state)
    
    # Assert
    with subtests.test("Covariance is symmetric"):
        assert np.allclose(result.covariance, result.covariance.T)

    with subtests.test("Covariance is positive definite"):
        assert np.all(np.linalg.eigvals(result.covariance) > 0)
```

### Test DSL

If your test setup is complicated, consider creating some simple DSL to express your intent in a more readable way.

```python
# Good
from tests.dsl import check

result = model.predict(...)
check.is_spd(result.covariance, atol=1e-8)
    
# Bad
result = model.predict(...)

for obstacle_covariance in result.covariances:
    assert np.allclose(obstacle_covariance, obstacle_covariance.T, atol=1e-8)
    assert np.all(np.linalg.eigvals(obstacle_covariance) > 0)
```

### Test Doubles

**Do not use mocks.** Use stubs, fake implementations or real components instead. This helps keep the tests black-box and focused on the behavior of the component under test, rather than its internal implementation. It also makes the tests more robust to refactoring and implementation changes[<sup>1</sup>](https://martinfowler.com/articles/mocksArentStubs.html).

```python
from tests.dsl import stubs

# Good: Stub that returns predetermined values
model = stubs.DynamicalModel.returns(
    rollouts=expected_rollouts,
    when_control_inputs_are=inputs,
    and_initial_state_is=initial_state,
)
planner = Mppi(..., model=model)

# Good: Fake component that implements the same interface but with toy logic
model = SimpleLinearModel(...)
planner = Mppi(..., model=model)

# Bad: Mocking the model's behavior and hardcoding its specs into the test. The 
# specs of the DynamicalModel are implementation details in the context of this test, 
# since the SUT is the MPPI planner, not the model.
from unittest.mock import Mock

model = Mock(spec=DynamicalModel)
model.predict.return_value = expected_rollouts

planner = Mppi(..., model=model)
```

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

Try to avoid documenting things that you think are obvious, or in a way that does not add new information.

```python
# Good
class InputBatch:
    """Batch of control input sequences for multiple rollouts."""
    
    ...

    @property
    def horizon(self) -> int:
        """The number of time steps in a single control input sequence of this batch."""
        ...  

# Bad
class InputBatch:
    """An instance of the InputBatch class."""

    ...

    @property
    def horizon(self) -> int:
        """Returns the horizon of this InputBatch."""
        ...
```
