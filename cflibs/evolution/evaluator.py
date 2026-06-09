"""AST-level blocklist scanner for evolved candidate code.

The CF-LIBS evolution loop (hierarchical ES, LLM-driven perturbations) produces
candidate replacements for scoped regions of the inversion pipeline. The hard
project constraint is that the *shipped* algorithm must remain physics-only:
no neural networks, no trained models, no PLS/PCR, no learned features.

This module parses a candidate's source text with :mod:`ast` and rejects any
candidate that imports or references a forbidden library. Violations are
emitted as structured :class:`BlocklistViolation` records; the convenience
:func:`assert_physics_only` raises :class:`BlocklistViolationError` so the
evolution driver can set ``fitness = -inf`` and skip physics evaluation.

See beads CF-LIBS-improved-3fy3 for the full allow/deny specification.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Iterable, Literal

ViolationKind = Literal["import", "import_from", "attribute", "dynamic_import"]

# Prefixes that must never appear in evolved candidate code.
#
# Single-token entries forbid the top-level module (``import sklearn`` etc.).
# Dotted entries forbid a specific submodule while leaving the parent package
# available — the parent (e.g. ``jax``) is required for the physics path.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
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

# Callables that smuggle imports past static analysis. Evolved candidates
# have no legitimate reason to use them; any occurrence is flagged as a
# ``dynamic_import`` violation regardless of the argument.
DYNAMIC_IMPORT_CALLS: tuple[str, ...] = (
    "__import__",
    "importlib.import_module",
    "importlib.__import__",
)


@dataclass(frozen=True)
class BlocklistViolation:
    """A single forbidden reference located in candidate source."""

    module: str
    lineno: int
    col_offset: int
    kind: ViolationKind

    def format(self) -> str:
        return f"line {self.lineno}: forbidden {self.kind} of {self.module!r}"


class BlocklistViolationError(RuntimeError):
    """Raised when candidate code references a forbidden library."""

    def __init__(self, violations: Iterable[BlocklistViolation]):
        self.violations: list[BlocklistViolation] = list(violations)
        detail = "\n".join(f"  - {v.format()}" for v in self.violations)
        super().__init__("Candidate code contains forbidden imports/references:\n" + detail)


def _match_forbidden(name: str) -> str | None:
    """Return the matching forbidden prefix for ``name``, or ``None``."""
    for prefix in FORBIDDEN_PREFIXES:
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    return None


def _flatten_attribute(node: ast.AST) -> str | None:
    """Flatten an :class:`ast.Attribute` chain to a dotted name string.

    Returns ``None`` if the chain does not terminate in an :class:`ast.Name`
    (e.g. for subscripts, calls, or other non-bare attribute chains).
    """
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _scan_import(node: ast.Import) -> list[BlocklistViolation]:
    """Flag every forbidden alias in an ``import a, b`` statement."""
    violations: list[BlocklistViolation] = []
    for alias in node.names:
        if _match_forbidden(alias.name):
            violations.append(
                BlocklistViolation(
                    module=alias.name,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    kind="import",
                )
            )
    return violations


def _scan_import_from(node: ast.ImportFrom) -> BlocklistViolation | None:
    """Flag a ``from X import Y`` statement that references a forbidden name."""
    module = node.module or ""
    matched: str | None = module if _match_forbidden(module) else None
    if matched is None:
        for alias in node.names:
            combined = f"{module}.{alias.name}" if module else alias.name
            if _match_forbidden(combined):
                matched = combined
                break
    if matched is None:
        return None
    return BlocklistViolation(
        module=matched,
        lineno=node.lineno,
        col_offset=node.col_offset,
        kind="import_from",
    )


def _scan_attribute(node: ast.Attribute) -> BlocklistViolation | None:
    """Flag a forbidden attribute access chain (e.g. ``jax.nn.relu``)."""
    flat = _flatten_attribute(node)
    if flat and _match_forbidden(flat):
        return BlocklistViolation(
            module=flat,
            lineno=node.lineno,
            col_offset=node.col_offset,
            kind="attribute",
        )
    return None


def _scan_call(node: ast.Call) -> BlocklistViolation | None:
    """Flag dynamic-import smuggling via :data:`DYNAMIC_IMPORT_CALLS`.

    Detects ``__import__("sklearn")``, ``importlib.import_module("torch")``,
    etc. Evolved candidates never need runtime-resolved imports, so any
    occurrence is rejected regardless of the string argument.
    """
    func = node.func
    called: str | None = None
    if isinstance(func, ast.Name):
        called = func.id
    elif isinstance(func, ast.Attribute):
        called = _flatten_attribute(func)
    if called is not None and called in DYNAMIC_IMPORT_CALLS:
        return BlocklistViolation(
            module=called,
            lineno=node.lineno,
            col_offset=node.col_offset,
            kind="dynamic_import",
        )
    return None


def _scan_node(node: ast.AST) -> list[BlocklistViolation]:
    """Return every blocklist violation contributed by a single AST node."""
    if isinstance(node, ast.Import):
        return _scan_import(node)
    if isinstance(node, ast.ImportFrom):
        violation = _scan_import_from(node)
    elif isinstance(node, ast.Attribute):
        violation = _scan_attribute(node)
    elif isinstance(node, ast.Call):
        violation = _scan_call(node)
    else:
        return []
    return [] if violation is None else [violation]


def _dedupe_by_location(raw: Iterable[BlocklistViolation]) -> list[BlocklistViolation]:
    """Drop duplicate violations sharing a source location and kind.

    Deduplicates attribute-chain hits that fire at every nesting depth
    (e.g. walking ``jax.nn.relu`` visits both the outer and inner Attribute).
    """
    seen: set[tuple[int, int, str]] = set()
    deduped: list[BlocklistViolation] = []
    for v in raw:
        key = (v.lineno, v.col_offset, v.kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped


def scan_source(source: str) -> list[BlocklistViolation]:
    """Parse ``source`` and return every blocklist violation found.

    The scanner inspects:

    * :class:`ast.Import` statements — ``import sklearn``, ``import jax.nn``,
      ``import torch as t``.
    * :class:`ast.ImportFrom` statements — ``from sklearn.linear_model import
      Ridge``, ``from jax import nn``.
    * :class:`ast.Attribute` access chains — ``jax.nn.relu(x)``.
    * :class:`ast.Call` nodes that invoke :data:`DYNAMIC_IMPORT_CALLS`
      (e.g. ``__import__("sklearn")``, ``importlib.import_module("torch")``).

    Duplicate hits arising from walking nested attribute chains are
    de-duplicated by source location.
    """
    tree = ast.parse(source)
    raw: list[BlocklistViolation] = []
    for node in ast.walk(tree):
        raw.extend(_scan_node(node))
    return _dedupe_by_location(raw)


def assert_physics_only(source: str) -> None:
    """Raise :class:`BlocklistViolationError` if ``source`` violates the blocklist.

    Note: ``ast.parse(source)`` is called internally; malformed candidate code
    surfaces as :class:`SyntaxError`, not :class:`BlocklistViolationError`.
    Callers that need a single exception type should catch both.
    """
    violations = scan_source(source)
    if violations:
        raise BlocklistViolationError(violations)


def assert_benchmark_relevance(diff: str, exercised_files: set[str]) -> None:
    """Reject changes that do not touch any file exercised by the benchmark.

    This prevents opening PRs for changes that have zero effect on the
    fitness signal (e.g. editing a specialized identifier that is never
    called for the current benchmark dataset).

    Args:
        diff: The unified git diff of the candidate change.
        exercised_files: The set of file paths (relative to repo root) that
            were actually executed during the benchmark run.
    """

    # Extract touched files from a unified diff (e.g. '--- a/path/to/file')
    touched = set()
    for line in diff.splitlines():
        if line.startswith("--- a/"):
            touched.add(line[6:])
        elif line.startswith("+++ b/"):
            touched.add(line[6:])

    if not touched.intersection(exercised_files):
        raise RuntimeError(
            f"Candidate diff touches {touched}, but none of these files are "
            f"exercised by the current benchmark ({exercised_files}). "
            "The change will have zero effect on the fitness signal."
        )
