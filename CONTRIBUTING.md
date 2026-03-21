# Contributing to CF-LIBS

Thank you for your interest in contributing to CF-LIBS! This guide will help you get started.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Areas for Contribution](#areas-for-contribution)

---

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

```bash
just setup
```

This installs:
- The package in editable mode
- Development dependencies (pytest, black, mypy, ruff, ty)
- A consistent command surface through `just`

### Verify Setup

```bash
# Stable local quality gate
just check

# Full pytest suite
just test

# Experimental next-generation type check
just typecheck-ty
```

---

## Coding Standards

### Python Style

- **PEP 8**: Follow PEP 8 style guidelines
- **Line length**: 100 characters (configured in `pyproject.toml`)
- **Type hints**: Use type hints for all public functions
- **Docstrings**: Use NumPy-style docstrings for all public functions/classes

### Code Formatting

We currently use `black` for repo formatting commands:

```bash
just fmt
```

We are evaluating Ruff's formatter for a future repo-wide migration:

```bash
just fmt-ruff-check
```

### Linting

We use `ruff` for the stable lint gate:

```bash
just lint
just lint-fix  # Auto-fix issues
```

### Type Checking

We use `mypy` for the stable type-check gate:

```bash
just typecheck
```

We are also evaluating `ty` in advisory mode:

```bash
just typecheck-ty
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
just test

# Run specific test file
pytest tests/test_constants.py -v

# Run with coverage
pytest tests/ --cov=cflibs --cov-report=html

# Run the fast local slice used by `just check`
just test-fast
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

### Documentation Files

- **API Reference**: `docs/API_Reference.md` - Auto-generated or manually maintained
- **User Guide**: `docs/User_Guide.md` - User-facing documentation
- **Contributing**: `CONTRIBUTING.md` - This file
- **README**: `README.md` - Project overview

### Updating Documentation

When adding new features:
1. Update relevant docstrings
2. Add examples to User Guide if user-facing
3. Update API Reference
4. Add/update README if major feature

---

## Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:

```
Short summary (50 chars or less)

Longer explanation if needed. Explain what and why, not how.
- What changed
- Why it changed
- Any breaking changes
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
   black cflibs/
   ruff check cflibs/
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

---

Thank you for contributing to CF-LIBS! 🎉
