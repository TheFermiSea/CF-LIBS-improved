"""Acceptance criterion #4 — only the carve-out files may contain a literal
``try: import jax`` block in ``cflibs/``.

The shared decorator ``@jit_if_available`` (``cflibs.core.jax_runtime``) is the
approved way for physics modules to consume JAX optionally. New code that
re-introduces the legacy ``try: import jax`` pattern would silently bypass
the shared adapter, defeating T1-1.

This test AST-walks every ``cflibs/**/*.py`` and asserts the pattern only
appears in the §5 carve-out list. Run as part of the swarm gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Paths are evaluated relative to the repo root so the test is portable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CFLIBS_ROOT = _REPO_ROOT / "cflibs"


# ADR-0001 §5 / T1-1 spec §5 — these files keep their literal
# ``try: import jax`` because they pre-date or sit outside the host/kernel
# split (capability detection, plugin scaffolding, evolution evaluator).
ALLOWED_TRY_IMPORT_JAX: frozenset[str] = frozenset(
    {
        "cflibs/core/jax_runtime.py",
        "cflibs/core/platform_config.py",
        "cflibs/hpc/gpu_config.py",
        "cflibs/hpc/distributed_mcmc.py",
        "cflibs/benchmark/unified.py",
        "cflibs/evolution/evaluator.py",
        "cflibs/inversion/runtime/streaming.py",
        # identify/* JAX ports — deferred to follow-on (T1-1 spec §5)
        "cflibs/inversion/identify/alias.py",
        "cflibs/inversion/identify/comb.py",
        "cflibs/inversion/identify/correlation.py",
        "cflibs/inversion/identify/line_detection.py",
        "cflibs/inversion/identify/spectral_nnls.py",
        # owned by T1-3 / T1-6 / other beads (T1-1 spec §5)
        "cflibs/inversion/physics/boltzmann_jax.py",
        "cflibs/inversion/solve/iterative.py",
        "cflibs/inversion/solve/bayesian.py",
        "cflibs/inversion/solve/joint_optimizer.py",
        "cflibs/inversion/solve/coarse_to_fine.py",
        "cflibs/inversion/preprocess/deconvolution.py",
        "cflibs/inversion/common/pca.py",
    }
)


# T1-1 in-progress migration tail — these files will be removed from the
# allowlist as each commit in the §8 migration order lands. The presence
# of this list is itself a temporary scaffold; once empty, delete it.
_T1_1_IN_PROGRESS: frozenset[str] = frozenset(
    {
        "cflibs/instrument/convolution.py",
        "cflibs/instrument/model.py",
        "cflibs/manifold/generator.py",
        "cflibs/manifold/loader.py",
    }
)


def _has_try_import_jax(tree: ast.AST) -> bool:
    """Return True iff the AST contains a ``try`` block whose body imports JAX."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for child in node.body:
            if isinstance(child, ast.Import):
                if any(
                    alias.name == "jax" or alias.name.startswith("jax.") for alias in child.names
                ):
                    return True
            elif isinstance(child, ast.ImportFrom):
                if child.module == "jax" or (child.module or "").startswith("jax."):
                    return True
    return False


def _iter_python_files() -> list[Path]:
    return sorted(_CFLIBS_ROOT.rglob("*.py"))


def test_no_unauthorized_try_import_jax_in_cflibs():
    """T1-1 AC #4 — physics modules should consume JAX via
    :func:`jit_if_available`, not via inline ``try: import jax`` blocks.

    This is the post-migration regression guard. It enforces that every
    JAX-consuming module under ``cflibs/`` outside the §5 carve-out list
    has been ported to the shared decorator.
    """
    offenders: list[str] = []
    for path in _iter_python_files():
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            continue
        if not _has_try_import_jax(tree):
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel in ALLOWED_TRY_IMPORT_JAX or rel in _T1_1_IN_PROGRESS:
            continue
        offenders.append(rel)

    if offenders:
        pretty = "\n  ".join(offenders)
        pytest.fail(
            "Found unauthorized `try: import jax` blocks outside the §5 "
            f"carve-out list:\n  {pretty}\n\n"
            "Replace with `@jit_if_available` / `@vmap_if_available` from "
            "`cflibs.core.jax_runtime`, or add to ALLOWED_TRY_IMPORT_JAX "
            "with a justification."
        )


def test_kernels_modules_do_not_import_host():
    """A ``kernels.py`` next to a ``host.py`` must not import from its host.

    This prevents the obvious cycle ``host.py -> kernels.py -> host.py`` and
    keeps kernel modules pure-numerics.
    """
    for kernels_path in _CFLIBS_ROOT.rglob("kernels.py"):
        host_path = kernels_path.with_name("host.py")
        if not host_path.exists():
            continue
        source = kernels_path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        package = ".".join(kernels_path.relative_to(_CFLIBS_ROOT).with_suffix("").parts[:-1])
        host_module = f"cflibs.{package}.host" if package else "cflibs.host"
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == host_module:
                pytest.fail(
                    f"{kernels_path.relative_to(_REPO_ROOT)} imports from "
                    f"{host_module}; kernel modules must not depend on host."
                )
