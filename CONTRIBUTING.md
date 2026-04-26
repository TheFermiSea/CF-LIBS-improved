# Contributing to CF-LIBS

Thank you for your interest in contributing to CF-LIBS! This guide will help you get started.

## Code of Conduct

CF-LIBS is committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/CF-LIBS.git
   cd CF-LIBS
   ```
3. **Create a branch** for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### Install in Development Mode

CF-LIBS uses `uv` for environment management and `setuptools` for packaging.

**Base installation (all platforms):**
```bash
uv venv --python 3.12
pip install -e ".[dev]"
```

**Apple Silicon (Metal backend):**
```bash
uv pip install -e ".[local]"
```

**NVIDIA GPU cluster deployment:**
```bash
uv pip install -e ".[cluster]"
```

### Verify Setup

```bash
# Run quality gates (ruff, black, mypy, pytest)
ruff check cflibs/ tests/
black --check cflibs/
mypy cflibs/
pytest tests/ -v

# Force CPU backend (useful for CI/testing)
JAX_PLATFORMS=cpu pytest tests/ -v
```

---

## Coding Standards

### Python Style

- **PEP 8**: Follow PEP 8 style guidelines
- **Line length**: 100 characters (configured in `pyproject.toml`)
- **Type hints**: Use type hints for all public functions
- **Docstrings**: Use NumPy-style docstrings for all public functions/classes

### Physics-Only Constraint

The shipped CF-LIBS algorithm is physics-only. See [`docs/Evolution_Framework.md`](docs/Evolution_Framework.md) for the forbidden-library list, allowed primitives, and enforcement (Ruff TID251 + AST scanner). Before submitting a PR touching `cflibs/` production code, run `ruff check cflibs/` or `python -m cflibs.evolution <your_file.py>` to verify no banned imports slipped in.

### Code Formatting

Use Black for formatting:

```bash
black cflibs/
```

### Linting and Type Checking

```bash
ruff check cflibs/ tests/       # Linting (includes TID251 physics-only check)
ruff check --fix cflibs/        # Auto-fix issues
mypy cflibs/                    # Type checking
```

### Example Code Style

```python
from typing import List, Optional, Tuple
import numpy as np

def calculate_emissivity(
    transition: Transition,
    population: float,
    wavelength_nm: Optional[float] = None
) -> float:
    """
    Calculate spectral emissivity for a transition.
    
    Parameters
    ----------
    transition : Transition
        Atomic transition
    population : float
        Upper level population in cm^-3
    wavelength_nm : float, optional
        Wavelength in nm (defaults to transition wavelength)
    
    Returns
    -------
    float
        Spectral emissivity in W m^-3 nm^-1
    
    Notes
    -----
    The emissivity is calculated using:
    ε = (hc / 4πλ) * A_ki * n_k
    """
    if wavelength_nm is None:
        wavelength_nm = transition.wavelength_nm
    # ... implementation
```

---

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Test file names: `test_*.py`
- Test function names: `test_*`
- Use pytest fixtures for common setup

### Example Test

```python
import pytest
import numpy as np
from cflibs.core import constants

def test_boltzmann_constant():
    """Test that Boltzmann constant has correct value."""
    assert abs(constants.KB_EV - 8.617333262e-5) < 1e-10

def test_temperature_conversion():
    """Test temperature unit conversion."""
    from cflibs.core.units import convert_temperature
    
    T_k = 10000.0
    T_ev = convert_temperature(T_k, 'K', 'eV')
    T_k_round = convert_temperature(T_ev, 'eV', 'K')
    
    assert abs(T_k - T_k_round) < 1e-6
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_constants.py -v

# Run with coverage
pytest tests/ --cov=cflibs --cov-report=html

# Skip slow and database-dependent tests
pytest tests/ -m "not slow and not requires_db"

# Force CPU backend (for reproducibility)
JAX_PLATFORMS=cpu pytest tests/ -v
```

### Test Coverage

Aim for >80% test coverage for new code. Focus on:
- Public API functions
- Edge cases and error conditions
- Critical physics calculations

---

## Documentation

### Docstrings

Use NumPy-style docstrings:

```python
def function_name(param1: Type, param2: Type) -> ReturnType:
    """
    Brief description.
    
    Longer description if needed, explaining the function's purpose,
    algorithm, or important details.
    
    Parameters
    ----------
    param1 : Type
        Description of param1
    param2 : Type
        Description of param2
    
    Returns
    -------
    ReturnType
        Description of return value
    
    Raises
    ------
    ValueError
        When invalid input is provided
    
    Notes
    -----
    Additional notes, equations, or references.
    
    Examples
    --------
    >>> result = function_name(1, 2)
    >>> print(result)
    3
    """
```

When adding a user-facing feature: update docstrings, extend `docs/User_Guide.md` with an example, and link it from `docs/API_Reference.md`. For internal changes, NumPy-style docstrings are sufficient.

---

## Submitting Changes

### Commit Messages

Use short imperative summaries (50 chars or less):

```
feat: Add hierarchical element selection
fix: Correct Voigt profile gradient calculation
docs: Update Bayesian inference user guide
perf: Optimize element identification with caching
refactor: Reorganize inversion subpackages
```

Optional body for context:
```
Brief explanation if the change needs justification.
- What changed
- Why it changed
- Any breaking changes or important notes
```

### Pull Request Process

1. **Update your branch**:
   ```bash
   git checkout main
   git pull upstream main
   git checkout your-branch
   git rebase main
   ```

2. **Ensure tests pass**:
   ```bash
   pytest tests/ -v
   ```

3. **Ensure code quality**:
   ```bash
   ruff check cflibs/ tests/
   black --check cflibs/
   mypy cflibs/
   ```

4. **Create Pull Request**:
   - Clear title and description
   - Reference related issues
   - Include examples if user-facing
   - Update documentation

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] Type hints added
- [ ] No linter errors
- [ ] All tests pass
- [ ] Examples work (if applicable)

---

## Areas for Contribution

### High Priority

1. **Closure Modernization (CoDa)**:
   - Aitchison distance / simplex geometry for compositional quality metrics
   - ILR-transformed concentration space for Boltzmann fitting
   - Bayesian Model Averaging across closure variants (standard/matrix/oxide)

2. **Atomic Database Augmentation**:
   - STARK-B Stark broadening parameter ingestion
   - Expanded partition function coverage (Z=1-92, stages I-III)
   - A_ki transition probability uncertainties and NIST accuracy grades

3. **Real-Data Validation**:
   - ChemCam/SuperCam PDS data ingestion and parsing
   - End-to-end validation against labeled Mars spectra

### Medium Priority

1. **Database Backends**:
   - VALD data source via AtomicDataSource ABC
   - Kentucky Atomic Line List for forbidden/intercombination transitions
   - VAMDC-TAP unified query client

2. **Documentation**:
   - API reference for inversion module
   - Tutorial notebooks for Bayesian inference
   - Example workflows for streaming/temporal analysis

3. **Advanced Features**:
   - Dirichlet prior for Bayesian closure
   - Dark element residual category for missing-mass detection
   - Multi-zone plasma models

### Always Welcome

- Bug fixes
- Documentation improvements
- Code quality improvements
- Example notebooks
- Test cases

---

## Physics Guidelines

When contributing physics code:

1. **Document equations**: Include LaTeX equations in docstrings
2. **Reference sources**: Cite papers or textbooks
3. **Validate**: Compare with literature or benchmarks
4. **Units**: Be explicit about units in docstrings
5. **Assumptions**: Clearly state physical assumptions

Example:

```python
def saha_equation_ratio(
    ionization_potential_ev: float,
    temperature_ev: float,
    electron_density_cm3: float
) -> float:
    """
    Calculate ionization ratio using Saha equation.
    
    Calculates n_{z+1} / n_z using:
    
    .. math::
        \\frac{n_{z+1} n_e}{n_z} = \\frac{SAHA_CONST}{n_e} T^{1.5} 
        \\exp\\left(-\\frac{IP}{kT}\\right)
    
    Parameters
    ----------
    ionization_potential_ev : float
        Ionization potential in eV
    temperature_ev : float
        Electron temperature in eV
    electron_density_cm3 : float
        Electron density in cm^-3
    
    Returns
    -------
    float
        Ratio n_{z+1} / n_z
    
    References
    ----------
    .. [1] Saha, M. N. (1920). "On a Physical Theory of Stellar Spectra".
           Proceedings of the Royal Society of London.
    """
```

---

## Getting Help

- **Issues**: Open an issue on GitHub for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Email**: Contact maintainers for sensitive issues

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
