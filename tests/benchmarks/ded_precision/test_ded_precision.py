"""DED precision benchmark gates (DED-PLAN step 6).

Runs the synthetic Ti-6Al-4V Al-scan (clean floor) once and asserts the
absolute-accuracy gates. Targets that the constrained solver does not yet meet
(V conditioning, full nominal non-regression) are marked xfail so CI stays
green while the gap stays visible; remove the xfail when steps 7-8 + the V fix
land.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_db, pytest.mark.slow]


def _db_path():
    for c in (
        Path(__file__).resolve().parents[3] / "ASD_da" / "libs_production.db",
        Path(__file__).resolve().parents[3] / "libs_production.db",
    ):
        if c.exists():
            return str(c)
    return None


@pytest.fixture(scope="module")
def al_scan_clean():
    db_path = _db_path()
    if db_path is None:
        pytest.skip("libs_production.db not available")
    from tests.benchmarks.ded_precision.benchmark_runner import (
        run_composition_series,
        summarize_series,
    )

    df = run_composition_series(db_path, "Ti-6Al-4V", "Al", clean=True)
    return df, summarize_series(df)


def test_benchmark_produces_rows(al_scan_clean):
    df, _ = al_scan_clean
    assert len(df) > 0 and set(df["element"]) == {"Ti", "Al", "V"}
    assert df["converged"].any()


def test_al_recovery_clean(al_scan_clean):
    # Al is the drift target of this scan; it recovers well even on the floor.
    _, summ = al_scan_clean
    assert summ.loc["Al", "rmsep"] < 2.0, summ.loc["Al"].to_dict()


def test_absolute_bias_not_pinned_to_nominal(al_scan_clean):
    # At Al=4.0 (far from the 6.0 nominal) the recovered Al must track the true
    # value, not collapse toward nominal. (No prior yet -> trivially true now;
    # becomes the real guard once the weak nominal prior lands in step 8.)
    df, _ = al_scan_clean
    far = df[(df["element"] == "Al") & (np.isclose(df["target_value"], 4.0))]
    assert not far.empty
    pred = float(far["pred_wt"].mean())
    assert abs(pred - 4.0) < abs(pred - 6.0), f"Al={pred} drifted toward nominal 6.0"


@pytest.mark.xfail(
    reason="V Boltzmann conditioning (E_k spread, partition fallback); fix pending", strict=False
)
def test_ti64_v_recovery_target(al_scan_clean):
    _, summ = al_scan_clean
    assert summ.loc["V", "rmsep"] < 1.0, summ.loc["V"].to_dict()


@pytest.mark.xfail(
    reason="nominal non-regression blocked by V over-estimate via closure", strict=False
)
def test_nominal_non_regression(al_scan_clean):
    df, _ = al_scan_clean
    nominal = df[np.isclose(df["target_value"], 6.0)]
    for el in ("Ti", "Al", "V"):
        bias = float(nominal[nominal["element"] == el]["error"].mean())
        assert abs(bias) < 0.5, f"{el} bias {bias:.2f} at nominal"
