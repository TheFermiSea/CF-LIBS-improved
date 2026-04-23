"""Unit tests for the evolution-candidate blocklist scanner.

The scanner enforces the CF-LIBS project's hard no-ML-in-output constraint
on every evolved candidate. These tests pin down which source patterns are
accepted vs rejected, including adversarial variants (aliases, nested
imports, attribute chains) that an LLM might emit.
"""

from __future__ import annotations

import pytest

from cflibs.evolution.evaluator import (
    DYNAMIC_IMPORT_CALLS,
    BlocklistViolationError,
    FORBIDDEN_PREFIXES,
    assert_physics_only,
    scan_source,
)

# ---------------------------------------------------------------------------
# Allowed-source fixtures — must produce zero violations.
# ---------------------------------------------------------------------------

ALLOWED_SOURCES: list[str] = [
    "import numpy as np",
    "import scipy.optimize",
    "import jax",
    "import jax.numpy as jnp",
    "from jax import jit, vmap",
    "from scipy.optimize import nnls",
    "from numpy.linalg import lstsq",
    "import math, numpy as np",
    # A plausible physics-only candidate snippet.
    """
import numpy as np
from scipy.optimize import nnls

def solve(A, b):
    x, _ = nnls(A, b)
    return np.asarray(x)
""",
]


@pytest.mark.parametrize("source", ALLOWED_SOURCES)
def test_allowed_sources_pass(source: str) -> None:
    assert scan_source(source) == []
    # Must not raise.
    assert_physics_only(source)


# ---------------------------------------------------------------------------
# Forbidden single-token imports.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden_module",
    ["sklearn", "torch", "tensorflow", "keras", "flax", "equinox", "transformers"],
)
def test_forbidden_single_import_rejected(forbidden_module: str) -> None:
    source = f"import {forbidden_module}"
    violations = scan_source(source)
    assert len(violations) == 1
    assert violations[0].module == forbidden_module
    assert violations[0].kind == "import"
    with pytest.raises(BlocklistViolationError):
        assert_physics_only(source)


def test_aliased_forbidden_import_rejected() -> None:
    """``import torch as t`` must fail — the alias does not launder the module."""
    source = "import torch as t\nx = t.tensor([1])"
    violations = scan_source(source)
    # Expect at least the import hit; the attribute ``t.tensor`` aliases
    # through the rebound name so it won't be detected by AST alone
    # (and doesn't need to be — the import is already fatal).
    kinds = {v.kind for v in violations}
    modules = {v.module for v in violations}
    assert "import" in kinds
    assert "torch" in modules


def test_dotted_forbidden_import_rejected() -> None:
    source = "import sklearn.linear_model"
    violations = scan_source(source)
    assert any(v.kind == "import" and v.module == "sklearn.linear_model" for v in violations)


# ---------------------------------------------------------------------------
# Forbidden from-imports.
# ---------------------------------------------------------------------------


def test_from_import_with_forbidden_parent_rejected() -> None:
    source = "from sklearn.linear_model import Ridge"
    violations = scan_source(source)
    assert len(violations) == 1
    assert violations[0].kind == "import_from"
    # Match reports the forbidden prefix, which is ``sklearn`` (a single-token
    # entry in FORBIDDEN_PREFIXES).
    assert violations[0].module.startswith("sklearn")


def test_from_import_nested_forbidden_submodule_rejected() -> None:
    """``from jax import nn`` must fail — ``jax.nn`` is forbidden though ``jax`` is not."""
    source = "from jax import nn"
    violations = scan_source(source)
    assert len(violations) == 1
    assert violations[0].kind == "import_from"
    assert violations[0].module == "jax.nn"


def test_from_import_forbidden_with_allowed_parent() -> None:
    source = "from jax.nn import relu"
    violations = scan_source(source)
    assert len(violations) == 1
    assert violations[0].kind == "import_from"
    assert violations[0].module == "jax.nn"


def test_jax_parent_alone_is_allowed() -> None:
    """``import jax`` and ``from jax import jit`` must stay allowed."""
    assert scan_source("import jax") == []
    assert scan_source("from jax import jit, vmap") == []


# ---------------------------------------------------------------------------
# Attribute-chain usage.
# ---------------------------------------------------------------------------


def test_attribute_chain_forbidden_flagged() -> None:
    """``jax.nn.relu(x)`` in the body must be flagged even without a jax.nn import."""
    source = """
import jax
x = jax.nn.relu(0.0)
"""
    violations = scan_source(source)
    attr_hits = [v for v in violations if v.kind == "attribute"]
    assert len(attr_hits) == 1
    assert attr_hits[0].module.startswith("jax.nn")


def test_nested_attribute_chain_deduplicated() -> None:
    """Walking nested Attribute nodes must not emit duplicate violations."""
    source = "import jax\nx = jax.experimental.stax.Dense(16)\n"
    violations = scan_source(source)
    attr_hits = [v for v in violations if v.kind == "attribute"]
    # One logical violation per source occurrence.
    assert len(attr_hits) == 1
    assert attr_hits[0].module.startswith("jax.experimental.stax")


# ---------------------------------------------------------------------------
# Error plumbing.
# ---------------------------------------------------------------------------


def test_violation_formatting() -> None:
    source = "import torch"
    [v] = scan_source(source)
    formatted = v.format()
    assert "line 1" in formatted
    assert "torch" in formatted
    assert "import" in formatted


def test_assert_physics_only_raises_with_all_violations() -> None:
    source = """
import sklearn
from torch import nn
"""
    with pytest.raises(BlocklistViolationError) as excinfo:
        assert_physics_only(source)
    # Both violations must be surfaced on the exception.
    assert len(excinfo.value.violations) == 2
    modules = {v.module for v in excinfo.value.violations}
    assert any(m.startswith("sklearn") for m in modules)
    assert any(m.startswith("torch") for m in modules)


def test_forbidden_prefixes_complete() -> None:
    """Smoke test: all expected ML libs are in the blocklist."""
    required = {
        "sklearn",
        "torch",
        "tensorflow",
        "keras",
        "flax",
        "equinox",
        "transformers",
        "jax.nn",
        "jax.experimental.stax",
    }
    assert required.issubset(set(FORBIDDEN_PREFIXES))


# ---------------------------------------------------------------------------
# Dynamic-import smuggling.
# ---------------------------------------------------------------------------


def test_dunder_import_rejected_for_forbidden_module() -> None:
    source = '__import__("sklearn")'
    violations = scan_source(source)
    dyn = [v for v in violations if v.kind == "dynamic_import"]
    assert len(dyn) == 1
    assert dyn[0].module == "__import__"


def test_dunder_import_rejected_for_innocent_module_too() -> None:
    """Dynamic imports are banned regardless of target — evolved code has no need."""
    source = '__import__("numpy")'
    violations = scan_source(source)
    dyn = [v for v in violations if v.kind == "dynamic_import"]
    assert len(dyn) == 1


def test_importlib_import_module_rejected() -> None:
    source = 'import importlib\nimportlib.import_module("torch")'
    violations = scan_source(source)
    dyn = [v for v in violations if v.kind == "dynamic_import"]
    assert len(dyn) == 1
    assert dyn[0].module == "importlib.import_module"


def test_dynamic_import_calls_registry_covers_common_forms() -> None:
    assert "__import__" in DYNAMIC_IMPORT_CALLS
    assert "importlib.import_module" in DYNAMIC_IMPORT_CALLS


def test_assert_physics_only_raises_on_dynamic_import() -> None:
    with pytest.raises(BlocklistViolationError):
        assert_physics_only('__import__("sklearn")')
