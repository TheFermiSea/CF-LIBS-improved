"""Detect the stale xfail on the bayesian end-to-end test.

Followup to test-coverage audit (2026-05-20): the @pytest.mark.xfail
on tests/benchmark/test_jax_workflows.py::test_bayesian_predictor_runs_end_to_end
references closed bead CF-LIBS-improved-359q. When the underlying timing
issue is fixed and the xfail is removed, this guard test will fail and
remind us to also delete it.
"""

from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TARGET = _REPO_ROOT / "tests" / "benchmark" / "test_jax_workflows.py"


def test_stale_xfail_marker_still_documented():
    source = _TARGET.read_text(encoding="utf-8")
    has_marker = (
        "CF-LIBS-improved-359q" in source
        and "@pytest.mark.xfail" in source
    )
    assert has_marker, (
        "The stale xfail referencing CF-LIBS-improved-359q appears to have "
        "been removed from tests/benchmark/test_jax_workflows.py. Delete "
        "tests/benchmark/test_jax_workflows_xfail_check.py as well — it "
        "only exists to flag the stale marker."
    )
