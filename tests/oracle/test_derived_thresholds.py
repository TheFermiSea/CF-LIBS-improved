"""Conformance test: ``cflibs.inversion.physics.derived_thresholds`` vs the machine-verified
``CflibsFormal.ErrorBudget`` oracle.

The ``error-budget`` scenario in ``fixtures.json`` is emitted by the Lean Float mirror; each
threshold value + invariant instantiates a PROVEN theorem. This test asserts our Python
formulas reproduce the verified values (rtol=1e-6) AND satisfy the proven invariants, so any
drift of the shipped thresholds from the proof fails CI. Regenerate: `lake exe oracle-fixtures`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cflibs.inversion.physics import derived_thresholds as dt

RTOL = 1e-6
FIXTURES = json.loads((Path(__file__).parent / "fixtures.json").read_text())
EB = next(s for s in FIXTURES["scenarios"] if s["kind"] == "error-budget")
IN = EB["inputs"]
TH = EB["thresholds"]


def _approx(a: float, b: float, rtol: float = RTOL) -> bool:
    return abs(a - b) <= rtol * max(1.0, abs(b))


# ---- threshold values match the verified Lean Float exports (1:1, rtol=1e-6) ----
_VALUE_CASES = {
    "requiredEnergySpread": lambda: dt.required_energy_spread(IN["tauBeta"], IN["snr"], IN["n"]),
    "maxPerLineError": lambda: dt.max_per_line_error(IN["tauBeta"], IN["n"], IN["ssE"]),
    "requiredMinLinesStat": lambda: dt.required_min_lines(IN["tauBeta"], IN["snr"], IN["vPerLine"]),
    "slopeTargetFromTempRel": lambda: dt.slope_target_from_temp_rel(
        IN["relTtarget"], IN["kB"], IN["T"]
    ),
    "densityBudgetFromComposition": lambda: dt.density_budget_from_composition(
        IN["tauC"], IN["Shat"], IN["card"]
    ),
    "noiseGain": lambda: dt.ols_noise_gain(IN["ssE"]),
}


@pytest.mark.parametrize("field", list(_VALUE_CASES))
def test_threshold_value_matches_oracle(field):
    got, expected = _VALUE_CASES[field](), TH[field]
    assert _approx(got, expected), f"{field}: ours {got} != verified {expected}"


# ---- the PROVEN invariants (theorem-tagged checks) — the meaningful part ----
def test_energy_spread_threshold_is_tight():
    """slope_error_bound at the required spread == τβ — the threshold is TIGHT, not loose
    (ErrorBudget.requiredEnergySpread_sufficient)."""
    ss_e = dt.required_energy_spread(IN["tauBeta"], IN["snr"], IN["n"])
    assert _approx(dt.slope_error_bound(IN["snr"], IN["n"], ss_e), IN["tauBeta"])


def test_snr_threshold_is_tight():
    """slope_error_bound at the max tolerable per-line error == τβ (ErrorBudget.maxPerLineError_sufficient)."""
    snr_max = dt.max_per_line_error(IN["tauBeta"], IN["n"], IN["ssE"])
    assert _approx(dt.slope_error_bound(snr_max, IN["n"], IN["ssE"]), IN["tauBeta"])


def test_temp_rel_identity_exact():
    """slope_target_from_temp_rel · (kB·T) == relTtarget — the EXACT |ΔT|/T = kB·T·|Δβ| identity
    (ErrorBudget.temp_rel_error_eq)."""
    tb = dt.slope_target_from_temp_rel(IN["relTtarget"], IN["kB"], IN["T"])
    assert _approx(tb * (IN["kB"] * IN["T"]), IN["relTtarget"])


def test_composition_budget_exact():
    """(card+1)·density_budget == τC·Ŝ — the composition→density budget identity
    (ErrorBudget.composition_target_sufficient)."""
    delta = dt.density_budget_from_composition(IN["tauC"], IN["Shat"], IN["card"])
    assert _approx((IN["card"] + 1.0) * delta, IN["tauC"] * IN["Shat"])


# ---- NEGATIVE test (must fail on a real formula bug) ----
def test_negative_sqrt_n_bug_breaks_tightness():
    """The √n in the slope bound is load-bearing: a (common) `n`-instead-of-√n bug makes the
    energy-spread threshold NO LONGER tight, so the proven invariant would catch it. A test that
    only ever passes proves nothing."""
    ss_e = dt.required_energy_spread(IN["tauBeta"], IN["snr"], IN["n"])
    buggy_bound = IN["snr"] * IN["n"] / math.sqrt(ss_e)  # BUG: n instead of √n
    assert not _approx(
        buggy_bound, IN["tauBeta"]
    ), "a √n bug would slip through the tightness invariant — the conformance is not discriminating"


# ---- pipeline-facing wrappers compose the proven formulas correctly ----
def test_min_lines_wrapper_floors_and_monotone():
    """The derived min_lines wrapper honors the floor and is monotone: a TIGHTER temperature
    target needs at least as many lines."""
    loose = dt.min_lines_per_element_for(rel_t_target=0.10, snr=0.02, energy_var_ev2=1.5)
    tight = dt.min_lines_per_element_for(rel_t_target=0.01, snr=0.02, energy_var_ev2=1.5)
    assert loose >= 3 and tight >= loose, f"floor/monotonicity violated ({loose} -> {tight})"
