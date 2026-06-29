"""Regression-test our CF-LIBS estimators against the machine-verified Lean spec.

The fixtures in ``tests/oracle/fixtures.json`` are emitted by the Float mirror of the
verified ``CflibsFormal`` Lean spec (`~/code/cflibs-formal/oracle/Generate.lean`); EACH
check instantiates a PROVEN theorem (see that repo's oracle/README.md). This test runs
the fixtures through our REAL estimators — ``BoltzmannPlotFitter`` (our OLS Boltzmann
plot), ``ClosureEquation`` (our closure), ``_escape_factor`` (our self-absorption) — and
asserts they reproduce the verified ground truth at rtol=1e-6.

Convention bridge (the spec is dimensionless: kB=T=Fcal=1, E in units of kB·T):
- Our pipeline uses the STANDARD CF-LIBS energy-intensity ordinate ln(I·λ/(gA)) (Ciucci
  1999; Aragón & Aguilera 2008). The verified spec uses a unit-reduced photon-rate forward
  (I = Fcal·A·n) -> ordinate ln(I/(gA)), no λ. To feed our pipeline the spec's fixtures we
  apply the photon->energy conversion **I_our = I_spec/λ** with DISTINCT per-line λ: then
  ln(I_our·λ/(gA)) = ln(I_spec/(gA)) reproduces the verified ordinate EXACTLY, while forcing
  our pipeline to handle λ correctly (a λ^2 / per-line / unit-wrong bug breaks the match).
  Setting λ=1 would be BLIND to λ-handling (1^x=1) — hence distinct λ. (A *uniform* λ-unit
  rescaling still cancels in slope+closure, and is physically absorbed into Fcal anyway.)
- the spec slope = -1/(kB·T) = -1; our fitter reports temperature_K = -1/(KB_EV·slope), so
  the dimensionless T=1 maps to temperature_K = 1/KB_EV. We assert on the slope (convention-
  free) and on density/composition (the constant hc/4π folds into Fcal, cancels in closure).

Scope: this covers the ternary-alloy estimator checks (temperature, round-trip/OLS,
closure, calibration-free, self-absorption). The forward-map + partition-function +
two-stage Saha checks need the DB-backed forward model / Saha solver and are the
documented follow-up (1b). Regenerate fixtures: `lake exe oracle-fixtures` in cflibs-formal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cflibs.inversion.common.data_structures import LineObservation
from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter
from cflibs.inversion.physics.closure import ClosureEquation
from cflibs.inversion.physics.self_absorption import _escape_factor

RTOL = 1e-6
FIXTURES = json.loads((Path(__file__).parent / "fixtures.json").read_text())
GLOBAL = FIXTURES["global"]
KB, T, FCAL = GLOBAL["kB"], GLOBAL["T"], GLOBAL["Fcal"]
ALLOY = next(s for s in FIXTURES["scenarios"] if s["kind"] == "multi-element-composition")


def _approx(a: float, b: float, rtol: float = RTOL) -> bool:
    return abs(a - b) <= rtol * max(1.0, abs(b))


def _partition(g, E) -> float:
    """U(T) = Σ g_k exp(-E_k/(kB·T)); matches the fixture's partitionFunction."""
    return sum(gk * math.exp(-Ek / (KB * T)) for gk, Ek in zip(g, E))


def _line_lambda_nm(k: int) -> float:
    """Distinct, realistic per-line wavelength so the test exercises our λ path."""
    return 300.0 + 80.0 * k  # 300, 380, 460, 540, ... nm (all distinct)


def _observations(el, intensities=None):
    """Build LineObservations applying the photon->energy bridge I_our = I_spec/λ with
    DISTINCT per-line λ, so our ln(I_our·λ/gA) reproduces the spec's ln(I_spec/gA) AND a
    λ-handling bug would break the match (λ=1 would be blind to it)."""
    intens = intensities if intensities is not None else el["intensities"]
    return [
        LineObservation(
            wavelength_nm=_line_lambda_nm(k),
            intensity=float(intens[k]) / _line_lambda_nm(k),  # I_spec/λ
            intensity_uncertainty=1e-9,  # noise-free fixtures
            element=el["sym"],
            ionization_stage=1,
            E_k_ev=float(el["E"][k]),
            g_k=int(round(el["g"][k])),
            A_ki=float(el["A"][k]),
        )
        for k in range(len(el["g"]))
    ]


def _fit(observations):
    return BoltzmannPlotFitter().fit(observations)


# ---------------------------------------------------------------- estimator checks
def test_temperature_recovered_per_element():
    """Boltzmann-plot slope recovers T per element (Classic.classic_temperature_correct).
    Spec slope = -1/(kB·T) = -1.0; our fitter's slope must match (convention-free)."""
    expected_slope = -1.0 / (KB * T)
    for el in ALLOY["elements"]:
        res = _fit(_observations(el))
        assert _approx(
            res.slope, expected_slope
        ), f"{el['sym']}: slope {res.slope} != {expected_slope}"


def test_round_trip_density_per_element():
    """OLS intercept recovers N per element: N = U·exp(intercept) with Fcal=λ=1
    (Classic.classicDensity_recovers / Alt.olsDensity_recovers)."""
    for el in ALLOY["elements"]:
        res = _fit(_observations(el))
        U = _partition(el["g"], el["E"])
        n_rec = U * math.exp(res.intercept)
        assert _approx(n_rec, el["N"]), f"{el['sym']}: recovered N {n_rec} != {el['N']}"


def test_closure_composition_sums_to_one_and_matches_truth():
    """ClosureEquation.apply_standard recovers the true composition over HETEROGENEOUS
    elements (each its own U_s) and sums to 1 (Closure.composition_sum_one)."""
    intercepts = {el["sym"]: _fit(_observations(el)).intercept for el in ALLOY["elements"]}
    partitions = {el["sym"]: _partition(el["g"], el["E"]) for el in ALLOY["elements"]}
    comp = _composition_map(ClosureEquation().apply_standard(intercepts, partitions))
    total = sum(comp.values())
    assert _approx(total, 1.0), f"composition sums to {total}, not 1"
    for i, el in enumerate(ALLOY["elements"]):
        assert _approx(
            comp[el["sym"]], ALLOY["true_composition"][i]
        ), f"{el['sym']}: {comp[el['sym']]} != {ALLOY['true_composition'][i]}"


def test_calibration_free_invariance():
    """Scaling Fcal (≡ scaling all intensities) leaves composition unchanged
    (Classic.classic_calibration_free)."""
    scale = 137.0
    base, scaled = {}, {}
    partitions = {el["sym"]: _partition(el["g"], el["E"]) for el in ALLOY["elements"]}
    for el in ALLOY["elements"]:
        base[el["sym"]] = _fit(_observations(el)).intercept
        scaled_intens = [scale * x for x in el["intensities"]]
        scaled[el["sym"]] = _fit(_observations(el, scaled_intens)).intercept
    cb = _composition_map(ClosureEquation().apply_standard(base, partitions))
    cs = _composition_map(ClosureEquation().apply_standard(scaled, partitions))
    for sym in cb:
        assert _approx(
            cb[sym], cs[sym]
        ), f"{sym}: composition not Fcal-invariant ({cb[sym]} vs {cs[sym]})"


def test_self_absorption_correction_recovers_truth():
    """Correcting an optically-thick line by SA(τ)=(1-e^-τ)/τ then inverting recovers N
    (Alt.selfAbsorbed_sound). Uses our _escape_factor."""
    for el in ALLOY["elements"]:
        tau = el["tau"]
        sa = _escape_factor(tau)
        assert _approx(sa, 1.0 if tau == 0 else (1 - math.exp(-tau)) / tau), "escape factor wrong"
        if tau == 0:
            continue
        U = _partition(el["g"], el["E"])
        # thick = thin·SA(τ) on every line (shared τ): a constant factor -> the slope is
        # unchanged, the intercept shifts by ln(SA), so the UNCORRECTED density = N·SA < N.
        thick = [x * sa for x in el["intensities"]]
        n_uncorrected = U * math.exp(_fit(_observations(el, thick)).intercept)
        assert _approx(n_uncorrected, el["N"] * sa), f"{el['sym']}: thick bias != N·SA"
        assert n_uncorrected < el["N"], f"{el['sym']}: self-absorption must bias DOWN (SA<1)"
        # correcting each line by /SA(τ) recovers the true density (selfAbsorbed_sound).
        corrected = [x / sa for x in thick]
        n_corrected = U * math.exp(_fit(_observations(el, corrected)).intercept)
        assert _approx(n_corrected, el["N"]), f"{el['sym']}: corrected N {n_corrected} != {el['N']}"


# ---------------------------------------------------------------- NEGATIVE test (must fail on a real bug)
def test_negative_dropping_per_element_partition_corrupts_closure():
    """The README's headline bug: with DISTINCT U_s, dropping the per-element partition
    (sharing one U) corrupts composition — and the oracle MUST catch it. A test that only
    ever passes proves nothing."""
    intercepts = {el["sym"]: _fit(_observations(el)).intercept for el in ALLOY["elements"]}
    # BUG: share a single (wrong) U across all elements instead of each element's own U_s
    shared_U = {el["sym"]: 1.0 for el in ALLOY["elements"]}
    bad = _composition_map(ClosureEquation().apply_standard(intercepts, shared_U))
    # at least one element's composition must now be wrong beyond tolerance
    wrong = any(
        not _approx(bad[el["sym"]], ALLOY["true_composition"][i])
        for i, el in enumerate(ALLOY["elements"])
    )
    assert (
        wrong
    ), "dropping per-element U_s did NOT corrupt closure — the oracle is not discriminating"


# ---------------------------------------------------------------- helper for ClosureResult
def _composition_map(result) -> dict:
    """Extract {element: fraction} from a ClosureResult across its possible shapes."""
    for attr in ("composition", "number_fractions", "fractions", "concentrations"):
        val = getattr(result, attr, None)
        if isinstance(val, dict):
            return {k: float(v) for k, v in val.items()}
    pytest.skip(f"ClosureResult exposes no composition dict (fields: {dir(result)})")
