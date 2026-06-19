# CF-LIBS Test Suite

Comprehensive test suite for CF-LIBS library.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_constants.py        # Core constants tests
├── test_units.py            # Unit conversion tests
├── test_config.py          # Configuration management tests
├── test_logging.py          # Logging tests
├── test_atomic.py           # Atomic data structures and database tests
├── test_plasma.py            # Plasma state and Saha-Boltzmann tests
├── test_radiation.py        # Radiation calculations tests
├── test_instrument.py       # Instrument modeling tests
├── test_io.py               # I/O utilities tests
├── test_echelle.py          # Echellogram extraction tests
├── test_spectrum_model.py   # SpectrumModel integration tests
├── test_cli.py              # CLI tests
└── test_integration.py      # End-to-end integration tests
```

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_plasma.py -v
```

### Run Specific Test

```bash
pytest tests/test_plasma.py::test_solve_plasma -v
```

### Run with Coverage

```bash
pytest tests/ --cov=cflibs --cov-report=html
```

### Run Only Fast Tests

```bash
pytest tests/ -m "not slow"
```

### Run Only Unit Tests

```bash
pytest tests/ -m "unit"
```

## Test Categories

### Unit Tests
- Test individual functions and classes in isolation
- Use mocks and fixtures for dependencies
- Fast execution

### Integration Tests
- Test complete workflows
- Use real database (temporary)
- May be slower

### Fixtures

Common fixtures available in `conftest.py`:
- `temp_db`: Temporary atomic database
- `atomic_db`: AtomicDatabase instance
- `sample_plasma`: Sample plasma state
- `sample_transition`: Sample atomic transition
- `sample_energy_level`: Sample energy level
- `sample_config_dict`: Sample configuration
- `temp_config_file`: Temporary config file

## Test Coverage Goals

- **Core modules**: >90% coverage
- **Physics modules**: >80% coverage
- **I/O and utilities**: >85% coverage
- **Overall**: >80% coverage

## Writing New Tests

### Test Naming

- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Example Test

```python
def test_function_name():
    """Test description."""
    # Arrange
    input_value = 10.0
    
    # Act
    result = function_to_test(input_value)
    
    # Assert
    assert result > 0
    assert result == pytest.approx(expected_value, rel=1e-6)
```

### Using Fixtures

```python
def test_with_fixture(atomic_db, sample_plasma):
    """Test using fixtures."""
    solver = SahaBoltzmannSolver(atomic_db)
    result = solver.solve_plasma(sample_plasma)
    assert result is not None
```

### Testing Exceptions

```python
def test_error_handling():
    """Test that errors are raised correctly."""
    with pytest.raises(ValueError, match="must be positive"):
        function_that_raises(-1.0)
```

### Gating tests on optional dependencies (JAX, etc.)

There are three mechanisms in the suite for skipping when an optional dependency
is missing. Use **exactly one** per module -- do not stack them:

- `@pytest.mark.requires_jax` (and `requires_bayesian`, `requires_uncertainty`,
  `requires_rust`): the **preferred** mechanism. The `pytest_collection_modifyitems`
  hook in `conftest.py` makes these markers functional -- it probes the dependency
  once at collection time and skips only the marked items. Use a module-level
  `pytestmark = pytest.mark.requires_jax` to gate the whole file.
- `jax = pytest.importorskip("jax")`: aborts the **whole module** at import time
  if JAX is absent. Use this *instead of* the marker only when you need the import
  to short-circuit before module-level JAX code runs.
- Manual `if <dep> is None: pytest.skip(...)` inside a test body: avoid -- it is the
  least discoverable form and is only justified for a per-test (not per-module)
  condition that the markers cannot express.

Do **not** combine `importorskip` + `requires_jax` + per-test manual skips in one
module (belt-and-suspenders). When `importorskip("jax")` runs, JAX is guaranteed
present, so any later `if not HAS_JAX...` guard is dead code.

## Continuous Integration

Tests are designed to run in CI environments:
- No external dependencies required (uses temporary databases)
- Fast execution (< 1 minute for full suite)
- Deterministic results
- Clear error messages

## Performance Tests

For performance-critical code, add benchmarks:

```python
def test_performance(benchmark):
    """Test performance of critical function."""
    result = benchmark(function_to_test, large_input)
    assert result is not None
```

Run with:
```bash
pytest tests/ --benchmark-only
```

