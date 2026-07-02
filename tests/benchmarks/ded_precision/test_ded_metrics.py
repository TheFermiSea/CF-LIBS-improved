"""DB-free unit tests for the DED metrics + solver-runner helpers (step 5)."""

import pytest

from tests.benchmarks.ded_precision.metrics import (
    absolute_metrics,
    log_ratio_metrics,
    min_detectable_change,
    precision_std,
    ratio_rmsep,
    renorm_to_set,
)
from tests.benchmarks.ded_precision.solver_runner import make_ne_diagnostic

pytestmark = pytest.mark.unit


def test_renorm_to_set_sums_100():
    out = renorm_to_set({"Ti": 45.0, "Al": 3.0, "V": 2.0, "O": 50.0}, ["Ti", "Al", "V"])
    assert sum(out.values()) == pytest.approx(100.0)
    assert out["Ti"] / out["Al"] == pytest.approx(15.0)  # ratio preserved


def test_absolute_metrics_perfect_and_biased():
    els = ["Ti", "Al", "V"]
    truth = [{"Ti": 90, "Al": 6, "V": 4}, {"Ti": 88, "Al": 8, "V": 4}]
    m = absolute_metrics(truth, truth, els)
    assert m["rmsep_joint"] == pytest.approx(0.0)
    pred = [{"Ti": 91, "Al": 6, "V": 4}, {"Ti": 89, "Al": 8, "V": 4}]  # Ti +1 both
    m2 = absolute_metrics(pred, truth, els)
    assert m2["per_element"]["Ti"]["bias"] == pytest.approx(1.0)
    assert m2["per_element"]["Ti"]["rmsep"] == pytest.approx(1.0)
    assert m2["per_element"]["Al"]["rmsep"] == pytest.approx(0.0)


def test_ratio_rmsep_zero_when_exact():
    truth = [{"Ti": 90, "Al": 6, "V": 4}]
    assert ratio_rmsep(truth, truth, "Al", "Ti") == pytest.approx(0.0)


def test_log_ratio_metrics_zero_when_exact():
    truth = [{"Ti": 90, "Al": 6, "V": 4}, {"Ti": 88, "Al": 8, "V": 4}]
    m = log_ratio_metrics(truth, truth, "Ti")
    assert m["reference"] == "Ti"
    assert m["per_numerator"]["V"]["rmsep"] == pytest.approx(0.0)
    assert m["per_numerator"]["Al"]["rmsep"] == pytest.approx(0.0)


def test_log_ratio_invariant_to_third_element_slosh():
    # Absolute wt% sloshes when a spurious element enters the closure, but the
    # V/Ti and Al/Ti log-ratios are invariant (the shared denominator cancels).
    import math

    truth = [{"Ti": 90, "Al": 6, "V": 4}]
    # Same physics, but closure now also carries 50 wt% of a contaminant O:
    # renormalize the metals to 50 wt% -> every absolute fraction halves.
    contaminated = [{"Ti": 45, "Al": 3, "V": 2, "O": 50}]
    # Absolute V RMSEP is large (4 -> 2 = 2 wt% slosh):
    from tests.benchmarks.ded_precision.metrics import absolute_metrics

    abs_m = absolute_metrics(contaminated, truth, ["Ti", "Al", "V"])
    assert abs_m["per_element"]["V"]["rmsep"] == pytest.approx(2.0)
    # Log-ratio error is ~0 (denominator cancels):
    lr_m = log_ratio_metrics(contaminated, truth, "Ti")
    assert lr_m["per_numerator"]["V"]["rmsep"] == pytest.approx(0.0, abs=1e-12)
    assert lr_m["per_numerator"]["Al"]["rmsep"] == pytest.approx(0.0, abs=1e-12)
    assert not math.isnan(lr_m["per_numerator"]["V"]["rmsep"])


def test_precision_std():
    reps = [{"Al": 6.0}, {"Al": 6.2}, {"Al": 5.8}]
    s = precision_std(reps, ["Al"])
    assert s["Al"] > 0


def test_min_detectable_change_scales_with_sqrt_n():
    assert min_detectable_change(1.0, 1) == pytest.approx(3.0)
    assert min_detectable_change(1.0, 100) == pytest.approx(0.3)


def test_make_ne_diagnostic_recovers_target():
    # measured = w_ref*(ne/1e17)^alpha so the ref width cancels -> exact ne
    for ne in (5e16, 1e17, 3e17):
        d = make_ne_diagnostic(ne)
        implied = 1e17 * (d.measured_fwhm_nm / d.stark_w_ref_nm) ** (1.0 / d.stark_alpha)
        assert implied == pytest.approx(ne, rel=1e-9)
