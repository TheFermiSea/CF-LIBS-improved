# Evolution Framework

## Overview

The `cflibs.evolution` package provides tooling for an LLM-driven hierarchical Evolution-Strategies (ES) loop that optimizes the CF-LIBS algorithm's thresholds, weights, and structural parameters. It is part of the **optimization process** — not the shipped algorithm.

**Distinguishing feature:** A hard constraint enforced at two enforcement points (Ruff lint rule + AST scanner) ensures the shipped algorithm remains **physics-only**:

- **Allowed in shipped code:** Physics equations, partition functions, Saha-Boltzmann plasma solvers, signal processing (baseline removal, deconvolution), numpy/scipy/jax.numpy.
- **Forbidden in shipped code:** Neural networks, trained models, PLS/PCR, scikit-learn, PyTorch, TensorFlow, Keras, Flax, Equinox, any dynamic imports or attribute chains that bypass static analysis.

ML is permitted in the **optimization process** — LLM prompts, council synthesis, evolution drivers. It is never permitted in the code that is shipped.

## The Physics-Only Constraint

### Enforcement Points

**1. Ruff TID251 (lint-time)**

Ruff's `flake8-tidy-imports` rule checks every import statement in the codebase. Violations appear at lint time:

```toml
[tool.ruff.lint.flake8-tidy-imports.banned-api]
"sklearn".msg = "NO ML in shipped CF-LIBS algorithm. See CF-LIBS-improved-3fy3."
"torch".msg = "NO ML in shipped CF-LIBS algorithm. See CF-LIBS-improved-3fy3."
"tensorflow".msg = "NO ML in shipped CF-LIBS algorithm. See CF-LIBS-improved-3fy3."
"keras".msg = "NO ML in shipped CF-LIBS algorithm. See CF-LIBS-improved-3fy3."
"flax".msg = "NO ML in shipped CF-LIBS algorithm. See CF-LIBS-improved-3fy3."
"equinox".msg = "NO ML in shipped CF-LIBS algorithm. See CF-LIBS-improved-3fy3."
"transformers".msg = "NO ML in shipped CF-LIBS algorithm. See CF-LIBS-improved-3fy3."
"jax.nn".msg = "NO ML in shipped CF-LIBS algorithm. See CF-LIBS-improved-3fy3."
"jax.experimental.stax".msg = "NO ML in shipped CF-LIBS algorithm. See CF-LIBS-improved-3fy3."
```

This catches explicit imports but does not detect attribute chains (`jax.nn.relu(x)`) or dynamic imports (`__import__("sklearn")`).

**2. AST Scanner (runtime, evolution driver)**

Every candidate produced by the evolution loop is parsed with Python's `ast` module and scanned for forbidden references before physics evaluation. The scanner detects:

- `import` statements
- `from ... import` statements
- Attribute chains (`jax.nn.relu`, `torch.Tensor`)
- Dynamic imports (`__import__`, `importlib.import_module`)

Violations cause fitness to be set to `-inf` and the candidate is discarded immediately.

### Forbidden List

**Prefixes:** Single-token entries (e.g. `sklearn`, `torch`) forbid the entire module. Dotted entries (e.g. `jax.nn`) forbid a submodule while leaving the parent package available (JAX is required for autodiff of physics expressions).

```python
FORBIDDEN_PREFIXES = (
    "sklearn",
    "torch",
    "tensorflow",
    "keras",
    "flax",
    "equinox",
    "transformers",
    "jax.nn",
    "jax.experimental.stax",
)
```

**Dynamic import calls:** Caught regardless of argument:

```python
DYNAMIC_IMPORT_CALLS = (
    "__import__",
    "importlib.import_module",
    "importlib.__import__",
)
```

### Allowed Primitives

```
- numpy
- scipy (optimize.nnls, optimize.minimize, linalg, special)
- jax and jax.numpy (for autodiff / jit of physics expressions)
- math, dataclasses, typing, functools
- cflibs.plasma (Saha-Boltzmann, partition functions)
- cflibs.radiation (Voigt / Gaussian / Lorentzian profiles)
- cflibs.atomic (NIST line database lookups)
- cflibs.inversion.common (LineObservation, BoltzmannFitResult)
```

### Rationale

The CF-LIBS physics is well-grounded in first-principles spectral modeling. Introducing neural networks would:

1. Require training data with uncertain ground truth (bootstrapping problem)
2. Break interpretability and transfer across instruments/matrices
3. Lose the ability to model novel plasma regimes outside training distribution
4. Couple the algorithm to a specific cloud/cluster (LLM API dependency)

Evolution optimizes the algorithm within the physics-only constraint, allowing human scientists to understand and audit every change.

## Architecture

The evolution loop operates in four phases:

### Phase 1: Perturbation Generation

An LLM (e.g. 27B Scout) reads the current algorithm state, the physics-grounding preamble, and the measurement error metrics. It generates candidate mutations (parameter tweaks, weight adjustments, occasionally small algorithmic rewrites). Each candidate is wrapped in a function signature and returned as Python source text.

Timeout: configurable per batch (default 45 seconds).

### Phase 2: Evaluation

Each candidate is:

1. **AST-scanned** for physics-only constraint violations (see `scan_source()`)
2. **If valid**, inserted into the inversion pipeline and evaluated on the multi-dataset benchmark suite (aalto, chemcam, supercam, usgs, nist_steel)
3. **If invalid**, marked with fitness = -inf and skipped

Parallelism: configurable worker pool (default 8 workers, 5 seconds per evaluation).

### Phase 3: Council Synthesis

A second LLM reads the top-K candidates (ranked by fitness) and the error metrics. It proposes a conservative aggregation: which parameters should be kept, which reverted, which adjusted. The council output is merged into a single "next generation" state.

### Phase 4: Structural Mutation

Every N batches, a third LLM proposes a small structural change: a new if/else branch in line selection, a new weighting scheme in the Boltzmann fit, a new closure equation variant. The proposal is AST-scanned before physics evaluation.

## Module Reference

### `cflibs.evolution.evaluator`

**AST-level blocklist scanner for evolved candidate code.**

#### Classes

| Class | Purpose |
|-------|---------|
| `BlocklistViolation` | Structured record of a single forbidden reference (module, lineno, col_offset, kind) |
| `BlocklistViolationError` | Raised when `assert_physics_only()` detects violations |

#### Constants

| Constant | Type | Purpose |
|----------|------|---------|
| `FORBIDDEN_PREFIXES` | `tuple[str, ...]` | Tuple of module/submodule names forbidden in evolved code |
| `DYNAMIC_IMPORT_CALLS` | `tuple[str, ...]` | Function names that perform runtime import resolution |
| `ViolationKind` | Literal | Union of "import", "import_from", "attribute", "dynamic_import" |

#### Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `scan_source(source)` | `str -> list[BlocklistViolation]` | Parse source with `ast.parse()`, return all violations (de-duplicated by location) |
| `assert_physics_only(source)` | `str -> None` | Raise `BlocklistViolationError` if any violations found |

#### Example: Programmatic Use

```python
from cflibs.evolution.evaluator import scan_source, assert_physics_only, BlocklistViolationError

candidate_code = """
import jax.nn as nn
def evolved_fit(spectrum):
    return nn.relu(spectrum)
"""

violations = scan_source(candidate_code)
for v in violations:
    print(v.format())  # "line 1: forbidden import of 'jax.nn'"

# Or use the assertion form:
try:
    assert_physics_only(candidate_code)
except BlocklistViolationError as e:
    print(e)  # "Candidate code contains forbidden imports/references: ..."
```

### `cflibs.evolution.prompts`

**Physics-grounding string primitives for LLM-driven evolution prompts.**

#### Constants

| Constant | Type | Purpose |
|----------|------|---------|
| `ALLOWED_PRIMITIVES` | `tuple[str, ...]` | Tuple of module names and description strings shown to LLM |
| `PHYSICS_GROUNDING_PREAMBLE` | `str` | Template with placeholders for forbidden and allowed lists |

#### Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `render_preamble()` | `() -> str` | Substitute current forbidden/allowed lists into preamble template, return full physics-grounding text |

#### Physics-Grounding Preamble

The preamble is shown to every LLM prompt in the evolution loop:

```
You are proposing a change to a physics-only CF-LIBS algorithm.

HARD CONSTRAINT — NON-NEGOTIABLE:
The shipped algorithm must be physics-based only. You may NOT introduce
neural networks, trained models, PLS/PCR, learned embeddings, learned
features, or anything that requires a training phase on labelled spectra.

FORBIDDEN modules (any import, attribute access, or dynamic import of
these will be rejected at AST scan, fitness = -inf):
  - sklearn
  - torch
  - tensorflow
  - keras
  - flax
  - equinox
  - transformers
  - jax.nn
  - jax.experimental.stax

ALLOWED primitives for the evolved algorithm:
  - numpy
  - scipy (optimize.nnls, optimize.minimize, linalg, special)
  - jax and jax.numpy (for autodiff / jit of physics expressions)
  - math, dataclasses, typing, functools
  - cflibs.plasma (Saha-Boltzmann, partition functions)
  - cflibs.radiation (Voigt / Gaussian / Lorentzian profiles)
  - cflibs.atomic (NIST line database lookups)
  - cflibs.inversion.common (LineObservation, BoltzmannFitResult)

ML is permitted in the optimization process that produces your proposal
(you, the search loop, the council). It is NEVER permitted in the
candidate code you emit.

Candidates that violate this constraint are dropped before physics
evaluation — you do not get another attempt on a flagged candidate.
```

#### Example: Composing an LLM Prompt

```python
from cflibs.evolution.prompts import render_preamble

physics_constraint = render_preamble()

full_prompt = f"""{physics_constraint}

Given the current inversion algorithm with fitness {current_fitness}, propose
three parameter mutations that might improve performance on the chemcam matrix.
Focus on threshold adjustments in line selection and Boltzmann fitting.
"""

# Pass full_prompt to LLM API
```

### `cflibs.evolution.config`

**Runtime configuration for the evolution driver (batch sizes, timeouts, enforcement mode).**

#### Classes

| Class | Attributes | Purpose |
|-------|-----------|---------|
| `EvolutionDriverConfig` | See below | Immutable (frozen) dataclass capturing driver settings |

#### EvolutionDriverConfig Fields

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `perturbations_per_batch` | `int` | 16 | Candidates generated per batch (Phase 1) |
| `perturbation_timeout_s` | `float` | 45.0 | Wall-clock timeout per LLM call |
| `evaluation_workers` | `int` | 8 | Parallel workers for Phase 2; 0 = serial |
| `evaluation_timeout_s` | `float` | 5.0 | Wall-clock timeout per candidate evaluation |
| `structural_mutation_cadence` | `int` | 10 | Run Phase 4 every K batches |
| `enforcement_mode` | `"hard"` \| `"warn"` | `"hard"` | "hard" = fitness -inf on blocklist violation; "warn" = log and continue (dev only) |
| `fitness_weights` | `Mapping[str, float]` | aalto, chemcam, supercam, usgs, nist_steel @ 1.0 | Per-dataset fitness weights |
| `overfitting_penalty` | `float` | 0.5 | Multiplier on per-dataset variance (punish unbalanced performance) |
| `max_wallclock_hours` | `float` | 72.0 | Safety cap on total loop runtime |

#### Example: Creating a Config

```python
from cflibs.evolution.config import EvolutionDriverConfig

config = EvolutionDriverConfig(
    perturbations_per_batch=32,
    evaluation_workers=16,
    enforcement_mode="hard",
    fitness_weights={
        "aalto": 1.0,
        "chemcam": 1.5,  # prioritize ChemCam
        "supercam": 1.0,
        "usgs": 1.0,
        "nist_steel": 1.0,
    },
)

# All fields are validated in __post_init__
```

### `cflibs.evolution.__main__`

**Command-line entry point for the blocklist scanner.**

#### Invocation

```bash
python -m cflibs.evolution <file_or_stdin>
```

#### Examples

```bash
# Scan a single file
python -m cflibs.evolution candidate.py

# Scan multiple files
python -m cflibs.evolution file_a.py file_b.py

# Scan stdin
cat candidate.py | python -m cflibs.evolution -
```

#### Output

Violations are printed to stderr in `path:line: message` format (one per line), compatible with editor/CI integration:

```
candidate.py:5: forbidden import of 'jax.nn'
candidate.py:12: forbidden attribute of 'torch.Tensor'
```

Exit code: 0 on success (no violations), 1 if any violations found.

#### Example: CI Integration

```yaml
# .github/workflows/evolution-check.yml
- name: Check evolved candidates
  run: |
    python -m cflibs.evolution cflibs/evolution/evolved_candidates/*.py
```

## Usage Examples

### Checking a Candidate Programmatically

```python
from cflibs.evolution.evaluator import assert_physics_only

new_candidate = """
def evolved_line_selector(spectrum, wavelengths):
    import numpy as np
    from cflibs.inversion.physics import select_quality_lines
    
    # Physics-based logic
    baseline = np.median(spectrum)
    peaks = np.where(spectrum > baseline + 3 * np.std(spectrum))[0]
    return wavelengths[peaks]
"""

try:
    assert_physics_only(new_candidate)
    print("Candidate is physics-only. Ready for evaluation.")
except BlocklistViolationError as e:
    print(f"Rejected: {e}")
    # fitness = -inf, skip this candidate
```

### Pre-Commit Hook

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: evolution-blocklist
      name: Check evolved candidates
      entry: python -m cflibs.evolution
      language: system
      files: ^cflibs/evolution/evolved_candidates/
      pass_filenames: true
```

### CI Job (multi-file)

```bash
# Check all evolved candidates before merging
python -m cflibs.evolution $(find cflibs/evolution/evolved_candidates -name "*.py")
exit_code=$?

# Also run ruff TID251 check on entire codebase
ruff check cflibs/ --select=TID
exit_code=$(($exit_code + $?))

exit $exit_code
```

## Extending the Blocklist

To forbid a new library in evolved code:

1. **Update `cflibs/evolution/evaluator.py`:**

   ```python
   FORBIDDEN_PREFIXES: tuple[str, ...] = (
       # ... existing entries ...
       "my_new_ml_lib",
   )
   ```

2. **Update `pyproject.toml`:**

   ```toml
   [tool.ruff.lint.flake8-tidy-imports.banned-api]
   # ... existing entries ...
   "my_new_ml_lib".msg = "NO ML in shipped CF-LIBS algorithm. See CF-LIBS-improved-3fy3."
   ```

3. **Test the scanner:**

   ```python
   from cflibs.evolution.evaluator import scan_source
   
   test_code = "import my_new_ml_lib"
   violations = scan_source(test_code)
   assert violations[0].module == "my_new_ml_lib"
   ```

## Related

- **Ruff TID251 documentation:** https://docs.astral.sh/ruff/rules/flake8-tidy-imports/
- **Evolution epic:** `CF-LIBS-improved-3fy3` (beads issue tracker)
- **Physics-only constraint rationale:** Physics-only algorithm is interpretable, auditable, and transfers across instruments and plasma regimes.
