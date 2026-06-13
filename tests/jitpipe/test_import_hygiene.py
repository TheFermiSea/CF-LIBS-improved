"""J0 AC5 — ``cflibs.jitpipe`` import-boundary invariants.

Two grep/AST-based guards (mirroring ``tests/test_jax_import_hygiene.py`` and
the host/kernel split documented at ``cflibs/radiation/kernels.py:72-78``):

1. **No SQLite in kernels.** No module under ``cflibs/jitpipe/`` *except* the
   carve-out trio (``host.py`` / ``snapshot.py`` / ``parity.py``) may import
   ``sqlite3``, ``cflibs.atomic.database``, ``cflibs.io``, or
   ``cflibs.jitpipe.host``. This keeps the jit-traced stage kernels free of any
   live DB connection.
2. **Parallel implementation (ADR-0004 D1).** Nothing *outside*
   ``cflibs/jitpipe/`` may import ``cflibs.jitpipe`` — jitpipe is a parallel
   re-implementation that the reference pipeline never depends on.

Pure-AST + grep; no JAX or DB required.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CFLIBS_ROOT = _REPO_ROOT / "cflibs"
_JITPIPE_ROOT = _CFLIBS_ROOT / "jitpipe"

#: Only these jitpipe modules may touch impure SQLite-backed inputs.
_CARVE_OUT = frozenset({"host.py", "snapshot.py", "parity.py"})

#: Banned import targets for non-carve-out jitpipe modules.
_BANNED_MODULES = frozenset(
    {
        "sqlite3",
        "cflibs.atomic.database",
        "cflibs.io",
        "cflibs.jitpipe.host",
    }
)


def _banned_imports(tree: ast.AST) -> set[str]:
    """Return the set of banned module names imported anywhere in the AST."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name in _BANNED_MODULES or name.startswith("cflibs.io."):
                    found.add(name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in _BANNED_MODULES or mod == "cflibs.io" or mod.startswith("cflibs.io."):
                found.add(mod)
            # ``from cflibs.jitpipe import host`` / ``from cflibs.jitpipe.host import x``
            if mod == "cflibs.jitpipe.host" or (
                mod == "cflibs.jitpipe" and any(a.name == "host" for a in node.names)
            ):
                found.add("cflibs.jitpipe.host")
    return found


def test_no_sqlite_in_jitpipe_kernels():
    """AC5 part 1 — only host/snapshot/parity may import SQLite-backed code."""
    offenders: dict[str, set[str]] = {}
    for path in sorted(_JITPIPE_ROOT.glob("*.py")):
        if path.name in _CARVE_OUT:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        banned = _banned_imports(tree)
        if banned:
            offenders[path.name] = banned

    assert not offenders, (
        "jitpipe kernel modules must not import SQLite-backed code "
        f"(only {sorted(_CARVE_OUT)} may):\n"
        + "\n".join(f"  {name}: {sorted(mods)}" for name, mods in offenders.items())
    )


def test_carveout_modules_exist():
    """The carve-out trio must actually exist (guards against silent typos)."""
    for name in _CARVE_OUT:
        assert (_JITPIPE_ROOT / name).exists(), name


def _imports_jitpipe(tree: ast.AST) -> bool:
    """Return True iff the AST imports ``cflibs.jitpipe`` (any form)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                a.name == "cflibs.jitpipe" or a.name.startswith("cflibs.jitpipe.")
                for a in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "cflibs.jitpipe" or mod.startswith("cflibs.jitpipe."):
                return True
            # ``from cflibs import jitpipe``
            if mod == "cflibs" and any(a.name == "jitpipe" for a in node.names):
                return True
    return False


def test_nothing_outside_jitpipe_imports_jitpipe():
    """AC5 part 2 (ADR-0004 D1) — the reference codebase never imports jitpipe."""
    offenders: list[str] = []
    for path in _CFLIBS_ROOT.rglob("*.py"):
        # jitpipe may import itself.
        if _JITPIPE_ROOT in path.parents or path.parent == _JITPIPE_ROOT:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        if _imports_jitpipe(tree):
            offenders.append(path.relative_to(_REPO_ROOT).as_posix())

    assert not offenders, (
        "cflibs.jitpipe is a parallel implementation (ADR-0004 D1); nothing "
        "outside cflibs/jitpipe/ may import it. Offenders:\n  " + "\n  ".join(offenders)
    )
