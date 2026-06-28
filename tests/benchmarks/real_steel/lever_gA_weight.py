"""Lever gA-weight — atomic-data (A_ki) quality ablation on the real-steel gate.

INVESTIGATION B (atomic-data quality). The hypothesis: poorly-characterized
transition probabilities ``A_ki`` (low NIST accuracy grade) dominate the
Boltzmann slope unless the fit inverse-variance-weights each line by its
atomic-data uncertainty ``sigma(A_ki)/A_ki``. If that weighting were missing or
un-populated, the noisy D/E-graded lines would corrupt the fit (a real accuracy
loss). This lever AUDITS whether the weighting is wired+populated and MEASURES
its effect on the un-overfittable real-steel composition gate.

AUDIT RESULT (the weighting is already shipped, not a missing win)
-----------------------------------------------------------------
* ``ASD_da/libs_production.db`` populates ``aki_uncertainty`` for **100%** of
  aki-bearing lines of every steel/DED element (Fe..V), mapped from the NIST
  accuracy grade (A->0.03, B+->0.07, B->0.10, C->0.25, D->0.50, E->1.00; the
  E=1.00 bucket is the heuristic catch-all for lines NIST never graded).
* The value flows end-to-end: ``AtomicDatabase.get_transitions`` ->
  ``SelectedLine`` / ``LineSpec`` (``aki_uncertainty=...``) ->
  ``extract_line_intensities`` -> ``LineObservation.aki_uncertainty`` ->
  ``IterativeCFLIBSSolver(aki_uncertainty_weighting=True)`` (default ON), whose
  ``_line_y_uncertainty`` folds ``sigma(A_ki)/A_ki`` in quadrature into the
  per-line ``sigma_y`` and the common-slope fit weights ``1/sigma_y^2``
  (``solve/iterative.py`` ~L1455). ``BoltzmannPlotFitter._build_sigma_y`` does
  the same for the standalone fitter. So the **8.383 wt% shipped CD-SB-OPC
  baseline already runs with gA inverse-variance weighting on.**

ABLATION RESULT (the weighting is a NO-OP on real steel)
--------------------------------------------------------
Measured here over all 36 real-steel samples, weighting ON vs OFF gives a
**0.000 wt%** delta in ``rmsep_overall`` in BOTH the fixed-T (OPC) regime and
the free-T (slope-recovered) regime. Two compounding reasons:

1. The winning OPC pipeline FIXES the plasma T (Zhao optimal-T lever), so the
   slope — the quantity the weights most affect — is not fit from the data.
2. The shipped neutral-anchor selection picks only a few, mostly same-grade
   lines per element, and the per-element weight-cap (``boltzmann_weight_cap``
   x5 median) bounds the dynamic range; a near-uniform per-element weight set
   leaves the intercept (hence composition) unchanged.

GRADE-GATE VARIANT (dropping low-grade lines REGRESSES)
------------------------------------------------------
Hard-dropping D/E-graded lines before the fit (keeping only A/B+/B/C) makes the
real-steel ``rmsep_overall`` WORSE (fewer lines -> a less-constrained per-element
intercept / closure). Atomic-data quality is better handled by *down-weighting*
(already on, but a no-op here) than by *selection*.

CONCLUSION / RECOMMENDATION
---------------------------
* gA inverse-variance weighting: keep ON (it is the physically-correct default,
  costs nothing, and helps the free-T single-element Boltzmann-plot regime that
  other pipelines use), but it is **neutral** on the real-steel OPC gate — it
  does NOT beat 8.383 because it is already in that number.
* The only atomic-data lever with headroom is improving the A_ki **values**
  (not their uncertainties) for the D/E-dominated minors (Si 89% D/E, Al 84%,
  Ni 81% of aki lines) via a gated Kurucz/VALD gA-fill. But the OPC per-element
  ``F`` already absorbs the *systematic* per-element A_ki bias, so the expected
  gain is the residual per-LINE within-element scatter only — a larger DB
  project that must itself be benchmark-gated before promotion.

Run::

    PYTHONPATH=$PWD python tests/benchmarks/real_steel/lever_gA_weight.py
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from cflibs.inversion.common import LineObservation  # noqa: E402
from cflibs.inversion.pipeline import _number_to_mass_fractions  # noqa: E402
from cflibs.inversion.physics.line_selection import select_lines_by_policy  # noqa: E402
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver  # noqa: E402
from tests.benchmarks.ded_precision.line_extractor import (  # noqa: E402
    extract_line_intensities,
)
from tests.benchmarks.ded_precision.line_lists import LineSpec  # noqa: E402
from tests.benchmarks.ded_precision.solver_runner import (  # noqa: E402
    make_ne_diagnostic,
)
from tests.benchmarks.real_steel.harness import run_benchmark  # noqa: E402

#: aki_uncertainty values (sigma(A_ki)/A_ki) the NIST grade map assigns to the
#: grades we KEEP under the grade-gate variant (A, B+, B, C). The dropped grades
#: are D (0.50) and E (1.00, the un-graded heuristic catch-all).
KEEP_SIGMAS = {0.03, 0.07, 0.1, 0.25}

#: Shipped real-steel best (CD-SB matrix-ordinate OPC, honest held-out); the
#: reference this investigation is measured against. The number already includes
#: gA inverse-variance weighting (it is the IterativeCFLIBSSolver default).
SHIPPED_CDSB_RMSEP = 8.383


def _neutral_anchor_specs(db, wl, elements: Sequence[str]) -> List[LineSpec]:
    """Shipped neutral-anchor candidate lines (lever-L2 policy) for ``elements``."""
    window = (float(wl.min()), float(wl.max()))
    specs: List[LineSpec] = []
    for e in elements:
        for s in select_lines_by_policy(db, e, window, 8, policy="neutral_anchor"):
            specs.append(
                LineSpec(
                    element=s.element,
                    ionization_stage=int(s.ionization_stage),
                    wavelength_nm=float(s.wavelength_nm),
                    E_k_ev=float(s.E_k_ev),
                    g_k=float(s.g_k),
                    A_ki=float(s.A_ki),
                    aki_uncertainty=s.aki_uncertainty,
                    is_resonance=bool(s.is_resonance),
                )
            )
    return specs


def _grade_gate(obs: Sequence[LineObservation]) -> List[LineObservation]:
    """Drop D/E-graded lines (keep A/B+/B/C); revert per element if it falls < 2 lines."""
    from collections import defaultdict

    kept = defaultdict(list)
    full = defaultdict(list)
    for o in obs:
        full[o.element].append(o)
        u = o.aki_uncertainty
        if u is None or round(float(u), 3) in KEEP_SIGMAS:
            kept[o.element].append(o)
    out: List[LineObservation] = []
    for el, lst in full.items():
        out.extend(kept[el] if len(kept[el]) >= 2 else lst)
    return out


def _solve(
    db,
    obs: Sequence[LineObservation],
    *,
    weighting: bool,
    fixed_T: "float | None",
) -> Dict[str, float]:
    """Constrained Saha-Boltzmann solve with injected n_e and a gA-weighting toggle.

    Mirrors ``solver_runner.run_constrained_solver`` but exposes
    ``aki_uncertainty_weighting`` (the toggle the ablation flips) and an optional
    ``fixed_T`` (the OPC regime); returns recovered wt%.
    """
    solver = IterativeCFLIBSSolver(
        db,
        saha_boltzmann_graph=True,
        max_iterations=30,
        aki_uncertainty_weighting=weighting,
        fixed_temperature_K=fixed_T,
    )
    res = solver.solve(
        list(obs), closure_mode="standard", stark_diagnostics=[make_ne_diagnostic(1e17)]
    )
    if not res.mass_fractions:
        res.mass_fractions = _number_to_mass_fractions(res.concentrations)
    return {k: 100.0 * v for k, v in res.mass_fractions.items()}


def make_solve_fn(*, weighting: bool, fixed_T: "float | None", grade_gate: bool):
    """Build a ``solve_fn(db, wl, intensity, truth)`` for the real-steel gate harness."""

    def _fn(db, wl, intensity, truth):
        specs = _neutral_anchor_specs(db, wl, list(truth))
        if not specs:
            return {}
        obs = extract_line_intensities(wl, intensity, specs, instrument_fwhm_nm=0.2)
        if grade_gate:
            obs = _grade_gate(obs)
        if not obs:
            return {}
        return _solve(db, obs, weighting=weighting, fixed_T=fixed_T)

    return _fn


def run_ablation(
    db_path: str = "ASD_da/libs_production.db",
    *,
    fixed_T: float = 9000.0,
    limit: "int | None" = None,
) -> Dict[str, Dict[str, float]]:
    """Run the gA-weighting ablation across the 36-sample real-steel gate.

    Returns the per-variant score dicts for: fixed-T weighting on/off, fixed-T
    grade-gate, and free-T weighting on/off. The headline is whether weighting
    on vs off changes ``rmsep_overall`` (the no-op test) and whether grade-gating
    helps or hurts.
    """
    variants = {
        "fixedT_w_on": dict(weighting=True, fixed_T=fixed_T, grade_gate=False),
        "fixedT_w_off": dict(weighting=False, fixed_T=fixed_T, grade_gate=False),
        "fixedT_grade_gate": dict(weighting=True, fixed_T=fixed_T, grade_gate=True),
        "freeT_w_on": dict(weighting=True, fixed_T=None, grade_gate=False),
        "freeT_w_off": dict(weighting=False, fixed_T=None, grade_gate=False),
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, cfg in variants.items():
        out[name] = run_benchmark(make_solve_fn(**cfg), db_path=db_path, limit=limit)
        print(
            f"[{name}] rmsep_overall={out[name]['rmsep_overall']:.3f} "
            f"rmsep_Fe={out[name].get('rmsep_Fe', float('nan')):.3f}",
            flush=True,
        )
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", default="ASD_da/libs_production.db")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fixed-t", type=float, default=9000.0)
    a = ap.parse_args()
    os.environ.setdefault("REALSTEEL_QUIET", "1")
    res = run_ablation(db_path=a.db_path, fixed_T=a.fixed_t, limit=a.limit)
    print("\n=== gA-weighting ablation (real-steel, plain neutral-anchor path) ===")
    for name, sc in res.items():
        print(
            f"  {name}: overall={sc['rmsep_overall']:.3f} Fe={sc.get('rmsep_Fe', float('nan')):.3f}"
        )
    on = res["fixedT_w_on"]["rmsep_overall"]
    off = res["fixedT_w_off"]["rmsep_overall"]
    print(f"\n  fixed-T weighting delta (on - off) = {on - off:+.4f} wt%  (0 => no-op)")
    print(f"  shipped CD-SB-OPC reference (weighting already ON) = {SHIPPED_CDSB_RMSEP} wt%")
