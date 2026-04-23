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
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _match_forbidden(alias.name):
                    raw.append(
                        BlocklistViolation(
                            module=alias.name,
                            lineno=node.lineno,
                            col_offset=node.col_offset,
                            kind="import",
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            matched: str | None = module if _match_forbidden(module) else None
            if matched is None:
                for alias in node.names:
                    combined = f"{module}.{alias.name}" if module else alias.name
                    if _match_forbidden(combined):
                        matched = combined
                        break
            if matched is not None:
                raw.append(
                    BlocklistViolation(
                        module=matched,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        kind="import_from",
                    )
                )
        elif isinstance(node, ast.Attribute):
            flat = _flatten_attribute(node)
            if flat and _match_forbidden(flat):
                raw.append(
                    BlocklistViolation(
                        module=flat,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        kind="attribute",
                    )
                )
        elif isinstance(node, ast.Call):
            # Detect dynamic-import smuggling: __import__("sklearn"),
            # importlib.import_module("torch"), etc. Evolved candidates never
            # need runtime-resolved imports, so any occurrence is rejected
            # regardless of the string argument.
            func = node.func
            called: str | None = None
            if isinstance(func, ast.Name):
                called = func.id
            elif isinstance(func, ast.Attribute):
                called = _flatten_attribute(func)
            if called is not None and called in DYNAMIC_IMPORT_CALLS:
                raw.append(
                    BlocklistViolation(
                        module=called,
                        lineno=node.lineno,
                        col_offset=node.col_offset,
                        kind="dynamic_import",
                    )
                )

    # Deduplicate attribute-chain hits that fire at every nesting depth
    # (e.g. walking ``jax.nn.relu`` visits both the outer and inner Attribute).
    seen: set[tuple[int, int, str]] = set()
    deduped: list[BlocklistViolation] = []
    for v in raw:
        key = (v.lineno, v.col_offset, v.kind)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(v)
    return deduped


def assert_physics_only(source: str) -> None:
    """Raise :class:`BlocklistViolationError` if ``source`` violates the blocklist."""
    violations = scan_source(source)
    if violations:
        raise BlocklistViolationError(violations)
