# CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment pipeline for CF-LIBS.

## Overview

CF-LIBS uses GitHub Actions for CI/CD. The pipeline includes:
- Automated testing across multiple Python versions and platforms
- Code quality checks (linting, type checking, formatting)
- Test coverage reporting
- Package building and publishing
- Performance benchmarking
- Documentation building

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**Jobs:**

#### Test Job
- **Matrix**: Tests across Python 3.8, 3.9, 3.10, 3.11 on Ubuntu and macOS
- **Steps**:
  1. Checkout code
  2. Set up Python with pip caching
  3. Install dependencies (`pip install -e ".[dev]"`)
  4. Run tests with coverage (`pytest --cov`)
  5. Upload coverage to Codecov

#### Lint Job
- **Platform**: Ubuntu Latest
- **Steps**:
  1. Checkout code
  2. Set up Python 3.11
  3. Install linting tools (ruff, black, mypy)
  4. Run Ruff for linting
  5. Check code formatting with Black
  6. Run MyPy for type checking

#### Type Check Job
- **Platform**: Ubuntu Latest
- **Steps**:
  1. Checkout code
  2. Set up Python 3.11
  3. Install type checking dependencies
  4. Run MyPy with relaxed settings

#### Test Optional Dependencies Job
- **Matrix**: Tests with optional dependency groups (`jax`, `hdf5`, `all`)
- **Steps**:
  1. Checkout code
  2. Set up Python 3.11
  3. Install dependencies with optional extras
  4. Run tests (excluding manifold tests if dependencies unavailable)

#### Build Job
- **Platform**: Ubuntu Latest
- **Dependencies**: Requires `test` and `lint` jobs to pass
- **Steps**:
  1. Checkout code
  2. Set up Python 3.11
  3. Install build tools
  4. Build package (`python -m build`)
  5. Verify package installation

### 2. Documentation Workflow (`.github/workflows/docs.yml`)

**Triggers:**
- Push to `main` branch (when docs change)
- Manual workflow dispatch

**Steps**:
1. Checkout code
2. Set up Python 3.11
3. Install Sphinx and dependencies
4. Build documentation
5. Deploy to GitHub Pages (if on main branch)

### 3. Release Workflow (`.github/workflows/release.yml`)

**Triggers:**
- GitHub release published

**Steps**:
1. Checkout code
2. Set up Python 3.11
3. Install build tools (`build`, `twine`)
4. Build package
5. Publish to PyPI (requires `PYPI_API_TOKEN` secret)

### 4. Performance Workflow (`.github/workflows/performance.yml`)

**Triggers:**
- Push to `main` branch
- Pull requests to `main`
- Weekly schedule (Sundays)
- Manual workflow dispatch

**Steps**:
1. Checkout code
2. Set up Python 3.11
3. Install dependencies with pytest-benchmark
4. Run benchmarks
5. Upload benchmark results (if on main branch)

## Configuration

### Required Secrets

For PyPI publishing, set the following secret in GitHub:
- `PYPI_API_TOKEN`: PyPI API token for package publishing

### Optional Secrets

- `CODECOV_TOKEN`: Codecov token for enhanced coverage reporting (optional)

## Local Testing

You can test CI workflows locally using [act](https://github.com/nektos/act):

```bash
# Install act
brew install act  # macOS
# or download from https://github.com/nektos/act/releases

# Run CI workflow locally
act -j test

# Run specific job
act -j lint

# Run with specific Python version
act -j test --matrix python-version:3.11
```

## Running Tests Locally

Before pushing, run tests locally:

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=cflibs --cov-report=html

# Run linting
ruff check cflibs/ tests/
black --check cflibs/ tests/

# Run type checking
mypy cflibs/ --ignore-missing-imports
```

## Code Quality Standards

### Linting
- **Ruff**: Fast Python linter (replaces flake8, isort, etc.)
- **Black**: Code formatter (100 character line length)
- Configuration in `pyproject.toml`

### Type Checking
- **MyPy**: Static type checker
- Relaxed settings for gradual typing adoption
- Configuration in `pyproject.toml`

### Testing
- **pytest**: Test framework
- **pytest-cov**: Coverage reporting
- **pytest-xdist**: Parallel test execution
- Configuration in `pytest.ini`

## Coverage Goals

- **Target**: >80% code coverage
- **Critical modules**: >90% coverage
- Coverage reports uploaded to Codecov automatically

## Performance Benchmarks

Performance tests use `pytest-benchmark`:
- Run weekly to track performance regressions
- Results uploaded to GitHub Actions artifacts
- Compare against baseline on main branch

## Troubleshooting

### CI Failures

1. **Test Failures**: Check test output in Actions tab
2. **Linting Failures**: Run `ruff check` and `black` locally
3. **Type Check Failures**: Run `mypy` locally
4. **Build Failures**: Test build locally with `python -m build`

### Common Issues

**Import Errors**: Ensure all dependencies are in `setup.py` and `pyproject.toml`

**Platform-Specific Failures**: Test on multiple platforms locally or use Docker

**Coverage Decreases**: Review coverage report and add tests for uncovered code

## Best Practices

1. **Run tests locally** before pushing
2. **Fix linting issues** before committing
3. **Add tests** for new features
4. **Update documentation** when adding features
5. **Monitor CI** for failures and fix promptly
6. **Review coverage** reports regularly

## Future Enhancements

- [ ] Add Docker-based testing for reproducibility
- [ ] Add GPU testing for JAX-dependent code
- [ ] Add security scanning (bandit, safety)
- [ ] Add dependency vulnerability scanning
- [ ] Add automated dependency updates (Dependabot)
- [ ] Add pre-commit hooks configuration
- [ ] Add performance regression detection
- [ ] Add automated changelog generation

