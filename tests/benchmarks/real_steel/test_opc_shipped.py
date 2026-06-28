"""Shipped-API reproduction gate for the known-matrix OPC mode.

This is the CRITICAL validation for the OPC promotion (docs/research/
real-steel-opc-promotion.md): it runs the real-steel held-out gate THROUGH THE
SHIPPED, physics-only code — :func:`cflibs.inversion.physics.opc.calibrate_opc`,
:func:`cflibs.inversion.physics.opc.apply_opc`, and the shipped neutral-anchor
line-selection policy
(:func:`cflibs.inversion.physics.line_selection.select_lines_by_policy`) — rather
than the benchmark *lever* modules (``lever_l3b_robust_opc``, ``best_config_v2``)
that defined the algorithm. It asserts the shipped code reproduces the winning v2
held-out ``rmsep_overall`` (10.12 wt%) under the SAME a-priori conditioning rule
and honest held-out methodology, below the 11.0 wt% guard.

Honesty (structural, by construction):

* The OPC calibration is built by the shipped :func:`calibrate_opc`, which sees
  ONLY standards (their own spectra + own certified compositions). It cannot peek
  at the sample being scored.
* Leave-one-out: a sample that is itself a selected standard is scored with a
  calibration RE-DERIVED from the other standards only (``calibrate_opc`` over
  the leave-one-out subset). A non-selected sample never entered the calibration,
  so the full calibration is already held-out for it.
* :func:`apply_opc` rescales only the observation intensities; it never reads a
  recovered composition (no positive-feedback loop).

The recovery callback wraps the shipped Saha-Boltzmann solver
(``run_constrained_solver`` -> ``IterativeCFLIBSSolver``) on the shipped
neutral-anchor candidate lines, which is exactly the pipeline v2 calibrated and
applied — so a faithful reproduction here proves the promotion is sound.

Marked ``slow`` and skipped when the real-steel parquet is absent.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from tests.benchmarks.real_steel.harness import PARQUET, load_real_steel, score  # noqa: E402

# Guard just above the achieved v2 held-out rmsep_overall (~10.12 wt%); the
# shipped-API reproduction must clear it. Matches RMSEP_GUARD in the lever gate.
RMSEP_GUARD = 11.0
# Guard just above the achieved L5 thin_filter held-out rmsep_overall (9.561 wt%);
# the shipped optically-thin-filter OPC path must clear it.
RMSEP_GUARD_THIN = 9.7
# Guard just above the achieved L6 CD-SB held-out rmsep_overall (8.383 wt%); the
# shipped CD-SB matrix-ordinate OPC path must clear it (new real-steel best).
RMSEP_GUARD_CDSB = 8.5


def _selected_to_linespecs(selected_lines):
    """Convert shipped ``SelectedLine`` candidates to the extractor's ``LineSpec``."""
    from tests.benchmarks.ded_precision.line_lists import LineSpec

    return [
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
        for s in selected_lines
    ]


def _neutral_anchor_specs(db, wl, elements):
    """Shipped neutral-anchor candidate lines (lever-L2 policy) for ``elements``."""
    from cflibs.inversion.physics.line_selection import select_lines_by_policy

    window = (float(wl.min()), float(wl.max()))
    specs = []
    for e in elements:
        specs.extend(
            _selected_to_linespecs(
                select_lines_by_policy(db, e, window, 8, policy="neutral_anchor")
            )
        )
    return specs


def _build_standards(db, *, thin_filter: bool = False, cdsb: bool = False, cdsb_scale: float = 1.0):
    """Build a shipped :class:`Standard` per real-steel sample.

    Each standard's ``recover(T)`` runs the shipped neutral-anchor selection +
    extraction + constrained Saha-Boltzmann solver at fixed ``T`` (memoized), and
    its certified composition is the sample's renormalized truth. Returns
    ``(standards, samples)`` where ``samples`` preserves spectra for scoring.

    When ``thin_filter`` is True the shipped optically-thin line filter
    (:func:`cflibs.inversion.physics.opc.select_optically_thin_lines`) is applied
    to the extracted observations BEFORE the solve, so ``calibrate_opc`` derives
    the OPC ``F`` on the same (thin-filtered) pipeline the held-out solve applies
    it to (the honest decoupling: ``F`` then carries only the residual static
    sensitivity, not the per-spectrum self-absorption the filter already removed).

    When ``cdsb`` is True the shipped CD-SB matrix ordinate replacement
    (:func:`cflibs.inversion.physics.opc.apply_cdsb_matrix`) is applied at the
    recovery temperature ``T`` and the precomputed global scale ``cdsb_scale``
    BEFORE the solve, so the OPC ``F`` is re-derived on the CD-SB pipeline (the
    honest decoupling: ``F`` corrects only the static residual, never the
    per-spectrum self-absorption the CD-SB ordinate already encodes).
    """
    from cflibs.inversion.physics.opc import (
        Standard,
        StandardRecovery,
        apply_cdsb_matrix,
        select_optically_thin_lines,
    )
    from tests.benchmarks.ded_precision.line_extractor import extract_line_intensities
    from tests.benchmarks.ded_precision.solver_runner import recovered_wt, run_constrained_solver

    samples = list(load_real_steel())
    standards = []
    # The real-steel parquet stores a single shared ``sample_name`` ('1137') for
    # all 36 rows, so a unique per-row name is synthesized for provenance + the
    # leave-one-out key (the certified compositions genuinely differ per row).
    for idx, (sid, wl, inten, truth) in enumerate(samples):
        elements = list(truth)

        def _make(
            wl=wl,
            inten=inten,
            elements=elements,
            thin_filter=thin_filter,
            cdsb=cdsb,
            cdsb_scale=cdsb_scale,
        ):
            @lru_cache(maxsize=None)
            def _recover(T_K: float) -> StandardRecovery:
                specs = _neutral_anchor_specs(db, wl, elements)
                if not specs:
                    return StandardRecovery(composition={}, converged=False, degenerate=True)
                obs = extract_line_intensities(wl, inten, specs, instrument_fwhm_nm=0.2)
                if thin_filter:
                    obs = select_optically_thin_lines(wl, inten, obs)
                    if not obs:
                        return StandardRecovery(composition={}, converged=False, degenerate=True)
                if cdsb:
                    apply_cdsb_matrix(wl, inten, obs, float(T_K), float(cdsb_scale))
                res = run_constrained_solver(db, obs, 1e17, fixed_temperature_K=float(T_K))
                degenerate = (
                    float((res.quality_metrics or {}).get("degenerate_composition", 0.0)) >= 0.5
                )
                return StandardRecovery(
                    composition=recovered_wt(res),
                    converged=bool(res.converged),
                    degenerate=degenerate,
                )

            return _recover

        standards.append(Standard(name=f"{sid}#{idx}", certified=truth, recover=_make()))
    return standards, samples


def _solve_heldout(
    db, wl, inten, elements, calibration, *, thin_filter: bool = False, cdsb: bool = False
) -> Dict[str, float]:
    """Held-out solve: shipped neutral-anchor lines -> [thin filter | CD-SB] -> apply_opc -> solve.

    With ``thin_filter`` the shipped optically-thin filter drops the width-broadened
    (self-absorbed) matrix lines BEFORE the OPC ``F`` rescale, exactly as the
    ``run_pipeline`` OPC pre-solve does (and matching how ``F`` was derived).

    With ``cdsb`` the shipped CD-SB matrix ordinate replacement
    (:func:`cflibs.inversion.physics.opc.apply_cdsb_matrix`) replaces the thick
    matrix element's lines with their width-derived columnar density at the
    calibration's ``robust_T_K`` and ``cdsb_scale`` BEFORE the OPC ``F`` rescale --
    exactly the ``run_pipeline`` ``opc_cdsb_matrix`` pre-solve, and matching how
    ``F`` was re-derived on the CD-SB pipeline.
    """
    from cflibs.inversion.physics.opc import (
        apply_cdsb_matrix,
        apply_opc,
        select_optically_thin_lines,
    )
    from tests.benchmarks.ded_precision.line_extractor import extract_line_intensities
    from tests.benchmarks.ded_precision.solver_runner import recovered_wt, run_constrained_solver

    specs = _neutral_anchor_specs(db, wl, elements)
    if not specs:
        return {}
    obs = extract_line_intensities(wl, inten, specs, instrument_fwhm_nm=0.2)
    if thin_filter:
        obs = select_optically_thin_lines(wl, inten, obs)
        if not obs:
            return {}
    if cdsb:
        apply_cdsb_matrix(
            wl, inten, obs, float(calibration.robust_T_K), float(calibration.cdsb_scale)
        )
    apply_opc(obs, calibration)  # rescales intensity in place by F[element]
    res = run_constrained_solver(db, obs, 1e17, fixed_temperature_K=float(calibration.robust_T_K))
    return recovered_wt(res)


def run_shipped_opc_gate(
    db_path: str = "ASD_da/libs_production.db", *, thin_filter: bool = False
) -> Dict[str, object]:
    """Reproduce the held-out RMSEP through the shipped OPC API (honest LOO).

    With ``thin_filter=False`` (default) this reproduces the v2 plain-OPC result;
    with ``thin_filter=True`` it exercises the shipped optically-thin line filter
    (real-steel L5 winning lever) threaded through BOTH the ``calibrate_opc``
    F-derivation and the held-out solve. Returns the held-out score dict plus the
    full-calibration provenance.
    """
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.physics.opc import calibrate_opc

    db = AtomicDatabase(db_path)
    standards, samples = _build_standards(db, thin_filter=thin_filter)

    # Full calibration over ALL standards (shipped conditioning gate + robust T + geomean F).
    cal_full = calibrate_opc(standards)
    selected = set(cal_full.selected_standards)
    print(
        f"[shipped-opc] selected {len(selected)} standards, robust_T={cal_full.robust_T_K:.0f}K",
        flush=True,
    )

    # Cache leave-one-out calibrations for the selected standards (the only samples
    # that contributed to cal_full); non-selected samples use cal_full (held-out).
    loo_cache: Dict[str, object] = {}
    results: List = []
    for i, (std, (sid, wl, inten, truth)) in enumerate(zip(standards, samples)):
        name = std.name  # unique per row; matches calibrate_opc's selected_standards
        if name in selected:
            if name not in loo_cache:
                loo_cache[name] = calibrate_opc([s for s in standards if s.name != name])
            cal = loo_cache[name]
        else:
            cal = cal_full
        pred = _solve_heldout(db, wl, inten, list(truth), cal, thin_filter=thin_filter)
        results.append((truth, pred or {}))
        print(f"  [{i + 1}] sample {sid} done", flush=True)

    held = score(results)
    return {
        "robust_T_K": cal_full.robust_T_K,
        "n_selected": len(selected),
        "selected_standards": list(cal_full.selected_standards),
        "conditioning_rule": cal_full.conditioning_rule,
        "F": dict(cal_full.F),
        "thin_filter": thin_filter,
        "held_out": held,
    }


def _cdsb_geomean(vals) -> float:
    """Geometric mean of strictly-positive finite values (1.0 if none)."""
    import numpy as _np

    v = [float(x) for x in vals if _np.isfinite(x) and x > 0]
    return float(_np.exp(_np.mean(_np.log(v)))) if v else 1.0


def run_shipped_cdsb_gate(db_path: str = "ASD_da/libs_production.db") -> Dict[str, object]:
    """Reproduce the L6 CD-SB held-out RMSEP through the shipped OPC API (honest LOO).

    Mirrors :func:`...lever_l6_cdsb.run_l6` exactly but calls the SHIPPED CD-SB
    primitives (:func:`cflibs.inversion.physics.opc.cdsb_raw_ordinate`,
    ``cdsb_global_scale``, ``apply_cdsb_matrix``) + :func:`calibrate_opc`:

    1. **Uncorrected calibration** (lever steps 1-2): ``calibrate_opc`` over the
       plain (no-CD-SB) standards gives the selected standards (in-sample
       conditioning gate) and the robust ``T`` (mean of the selected standards'
       optimal ``T*``). Conditioning + ``T*`` are on the uncorrected pipeline, as
       in the lever.
    2. **Global CD-SB scale ``S``** (lever step 3): geometric mean over the
       selected standards of ``cdsb_global_scale`` at the robust ``T`` -- one
       constant for every spectrum, standards only.
    3. **OPC ``F`` on the CD-SB pipeline** (lever step 4): per selected standard,
       ``_derive_F`` on the CD-SB-handled recovery (CD-SB ordinate at ``S`` +
       robust ``T``); robust + leave-one-out geometric-mean ``F`` (clamp-saturated
       filtered).
    4. **Held-out solve** (lever step 5): per sample, the shipped CD-SB ordinate
       replacement + OPC ``F`` rescale + fixed-``T`` solve, scored with honest
       leave-one-out (a selected standard never sees its own ``F``).

    No parameter is tuned to held-out truth (the COG exponent 0.56 and the
    optically-thin reference width 0.08 nm are physical CD-SB constants). Returns
    the held-out score dict plus full-calibration provenance.
    """
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.physics.opc import (
        OPCCalibration,
        _derive_F,
        _geomean_F,
        calibrate_opc,
        cdsb_global_scale,
        cdsb_raw_ordinate,
    )
    from tests.benchmarks.ded_precision.line_extractor import extract_line_intensities

    db = AtomicDatabase(db_path)

    # 1. Uncorrected calibration -> selected standards + robust T (lever steps 1-2).
    std_unc, samples = _build_standards(db)
    cal0 = calibrate_opc(std_unc)
    robust_T = float(cal0.robust_T_K)
    selected = list(cal0.selected_standards)
    selected_set = set(selected)
    name_to_idx = {s.name: i for i, s in enumerate(std_unc)}
    print(
        f"[shipped-cdsb] selected {len(selected)} standards, robust_T={robust_T:.0f}K",
        flush=True,
    )

    # 2. Global CD-SB unit scale S over the selected standards at robust_T.
    S_list = []
    for name in selected:
        sid, wl, inten, truth = samples[name_to_idx[name]]
        specs = _neutral_anchor_specs(db, wl, list(truth))
        if not specs:
            continue
        obs = extract_line_intensities(wl, inten, specs, instrument_fwhm_nm=0.2)
        raw, _diag = cdsb_raw_ordinate(wl, inten, obs, robust_T)
        S_list.append(cdsb_global_scale(obs, raw))
    cdsb_scale = _cdsb_geomean(S_list)
    print(f"[shipped-cdsb] global CD-SB scale S = {cdsb_scale:.4g}", flush=True)

    # 3. Per-standard OPC F on the CD-SB pipeline (robust + leave-one-out).
    std_cdsb, _ = _build_standards(db, cdsb=True, cdsb_scale=cdsb_scale)
    cdsb_by_name = {s.name: s for s in std_cdsb}
    F_per_std = {name: _derive_F(cdsb_by_name[name], robust_T) for name in selected}
    robust_F = _geomean_F(list(F_per_std.values()))
    loo_F: Dict[str, Dict[str, float]] = {}
    for name in selected:
        others = [F_per_std[m] for m in selected if m != name]
        loo_F[name] = _geomean_F(others) if others else robust_F
    print(
        f"[shipped-cdsb] derived F on the CD-SB pipeline for {len(selected)} standards", flush=True
    )

    # 4. Apply robust (T, S, F) + CD-SB held-out and score (honest LOO per sample).
    rule = cal0.conditioning_rule
    results: List = []
    for i, (std, (sid, wl, inten, truth)) in enumerate(zip(std_unc, samples)):
        F = loo_F[std.name] if std.name in selected_set else robust_F
        cal = OPCCalibration(
            robust_T_K=robust_T,
            F=F,
            selected_standards=selected,
            conditioning_rule=rule,
            cdsb_scale=cdsb_scale,
        )
        pred = _solve_heldout(db, wl, inten, list(truth), cal, cdsb=True)
        results.append((truth, pred or {}))
        print(f"  [{i + 1}] sample {sid} done", flush=True)

    held = score(results)
    return {
        "robust_T_K": robust_T,
        "n_selected": len(selected),
        "selected_standards": selected,
        "conditioning_rule": rule,
        "cdsb_scale": float(cdsb_scale),
        "robust_F": {str(k): float(v) for k, v in robust_F.items()},
        "held_out": held,
    }


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(PARQUET), reason=f"real-steel parquet not present: {PARQUET}"
)
def test_shipped_opc_heldout_rmsep_below_guard():
    out = run_shipped_opc_gate()
    held = out["held_out"]
    overall = held["rmsep_overall"]
    assert np.isfinite(overall), f"shipped-OPC held-out RMSEP is not finite: {overall!r}"
    assert overall <= RMSEP_GUARD, (
        f"shipped-API OPC reproduction regressed: held-out rmsep_overall {overall:.3f} > "
        f"guard {RMSEP_GUARD} (v2 lever achieved 10.12). per-element: "
        + ", ".join(f"{k}={held[k]:.2f}" for k in sorted(held) if k.startswith("rmsep_"))
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(PARQUET), reason=f"real-steel parquet not present: {PARQUET}"
)
def test_shipped_opc_thin_filter_heldout_rmsep_below_guard():
    """Shipped optically-thin line filter (real-steel L5 lever) on the OPC path.

    Threads :func:`cflibs.inversion.physics.opc.select_optically_thin_lines` through
    BOTH the ``calibrate_opc`` F-derivation and the held-out solve (the same
    a-priori conditioning rule + honest leave-one-out as the plain-OPC gate, no
    held-out peeking). The lever achieved held-out rmsep_overall 9.561 wt%; this
    asserts the shipped path clears the 9.7 wt% guard just above it, improving on
    the plain-OPC 10.12 baseline.
    """
    out = run_shipped_opc_gate(thin_filter=True)
    held = out["held_out"]
    overall = held["rmsep_overall"]
    assert np.isfinite(overall), f"shipped-OPC thin-filter held-out RMSEP not finite: {overall!r}"
    assert overall <= RMSEP_GUARD_THIN, (
        f"shipped-API OPC thin-filter reproduction regressed: held-out rmsep_overall "
        f"{overall:.3f} > guard {RMSEP_GUARD_THIN} (L5 lever achieved 9.561). per-element: "
        + ", ".join(f"{k}={held[k]:.2f}" for k in sorted(held) if k.startswith("rmsep_"))
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(PARQUET), reason=f"real-steel parquet not present: {PARQUET}"
)
def test_shipped_cdsb_matrix_heldout_rmsep_below_guard():
    """Shipped CD-SB matrix-ordinate OPC path (real-steel L6 lever) held-out gate.

    Threads the shipped CD-SB primitives
    (:func:`cflibs.inversion.physics.opc.apply_cdsb_matrix` etc.) through BOTH the
    OPC ``F`` re-derivation and the held-out solve, with the SAME a-priori
    conditioning rule + honest leave-one-out as the plain-OPC gate (no held-out
    peeking, no parameter tuned to held-out truth). The L6 lever achieved held-out
    rmsep_overall 8.383 wt% (the new real-steel best, Fe 19.6 -> 16.5); this asserts
    the shipped path clears the 8.5 wt% guard just above it, improving on the L5
    thin-filter 9.561 and the plain-OPC 10.12 baselines.
    """
    out = run_shipped_cdsb_gate()
    held = out["held_out"]
    overall = held["rmsep_overall"]
    assert np.isfinite(overall), f"shipped-CD-SB held-out RMSEP not finite: {overall!r}"
    assert overall <= RMSEP_GUARD_CDSB, (
        f"shipped-API CD-SB reproduction regressed: held-out rmsep_overall "
        f"{overall:.3f} > guard {RMSEP_GUARD_CDSB} (L6 lever achieved 8.383). per-element: "
        + ", ".join(f"{k}={held[k]:.2f}" for k in sorted(held) if k.startswith("rmsep_"))
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--thin-filter", action="store_true")
    ap.add_argument("--cdsb", action="store_true", help="CD-SB matrix-ordinate mode (L6)")
    a = ap.parse_args()
    if a.cdsb:
        res = run_shipped_cdsb_gate()
        tag = "cdsb"
    else:
        res = run_shipped_opc_gate(thin_filter=a.thin_filter)
        tag = "thin_filter" if res["thin_filter"] else "plain"
    hs = res["held_out"]
    print(f"\n[shipped-opc:{tag}] real-steel held-out RMSEP (wt%):")
    print(f"  n_selected: {res['n_selected']} -> {res['selected_standards']}")
    print(f"  robust_T_K: {res['robust_T_K']:.0f}")
    print(f"  conditioning rule: {res['conditioning_rule']}")
    for k in sorted(hs):
        v = hs[k]
        print(f"    {k}: {v:.3f}" if isinstance(v, float) else f"    {k}: {v}")
