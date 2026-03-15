"""Regression tests for import-time safety of optional stacks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_import_snippet(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        if not existing_pythonpath
        else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_radiation_import_is_safe_without_forced_cpu_backend():
    """Importing cflibs.radiation should not fail at module import time."""
    result = _run_import_snippet("import cflibs.radiation; print('ok')")
    assert result.returncode == 0, result.stderr or result.stdout


def test_hybrid_import_is_safe_when_optional_stack_fails():
    """Optional Bayesian/JAX failures must not prevent hybrid imports."""
    result = _run_import_snippet("from cflibs.inversion.hybrid import HybridInverter; print('ok')")
    assert result.returncode == 0, result.stderr or result.stdout