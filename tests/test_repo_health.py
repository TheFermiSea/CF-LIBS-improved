"""Repository-health gates.

Currently enforces the ADR-0001 / T1-6 spec section 6 acceptance criterion
that every file inside the ``cflibs/inversion/solve/bayesian/`` package
remains under 800 LOC.
"""

from __future__ import annotations

import glob
import os


def _wc_l(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def test_bayesian_package_files_under_800_loc():
    """Every ``cflibs/inversion/solve/bayesian/*.py`` must be < 800 lines."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pattern = os.path.join(repo_root, "cflibs", "inversion", "solve", "bayesian", "*.py")
    files = sorted(glob.glob(pattern))
    assert files, f"No bayesian package files matched {pattern!r}"

    oversized = {path: _wc_l(path) for path in files if _wc_l(path) >= 800}
    assert not oversized, "T1-6 spec violation -- bayesian/*.py over 800 LOC: " + ", ".join(
        f"{os.path.basename(p)}={n}" for p, n in oversized.items()
    )
