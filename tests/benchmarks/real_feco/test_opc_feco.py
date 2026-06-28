"""Cross-matrix generality check: shipped known-matrix/OPC mode on real Fe-Co.

Runs the SAME shipped, physics-only code path validated on real steel
(:func:`cflibs.inversion.physics.opc.calibrate_opc` / ``apply_opc`` /
``select_optically_thin_lines`` + the neutral-anchor line-selection policy) on a
NEW alloy system -- the figshare 21984989 Fe-Co ladder (CEITEC, J. Vrabel; MIT) --
to test whether the real-steel OPC win (held-out RMSEP 39 -> 9.56 wt%, ~4x)
generalizes. Nothing here imports a benchmark *lever* module; every primitive is
the shipped API, exactly like ``tests/benchmarks/real_steel/test_opc_shipped.py``.

Two configs are scored with identical line selection (so the comparison isolates
OPC, not line picking):

* **baseline** -- plain calibration-free: neutral-anchor lines, solver recovers
  its own ``T``, no OPC ``F``. Uses no standards, so it is inherently held-out.
* **OPC (known-matrix)** -- shipped ``calibrate_opc`` (a-priori conditioning gate
  + robust ``T`` + geomean ``F``) on the Fe-Co ladder points, honest leave-one-out,
  ``apply_opc`` rescale, optional shipped optically-thin line filter.

Honesty is structural (same guarantees as the steel gate): ``calibrate_opc`` sees
only standards; a scored sample that is itself a selected standard is recalibrated
from the others (LOO); ``apply_opc`` never reads a recovered composition.

REGIME CAVEAT (read the result with this in mind): Fe-Co is a 2-element near-binary
ladder spanning the FULL 0-100 wt% range. The OPC known-matrix premise -- one
dominant matrix element + minors, a single shared per-element sensitivity ``F`` --
holds for steel but is only marginally applicable here: every Fe-Co sample is a
*different* matrix, so a single geomean ``F`` averages over wildly different
self-absorption regimes. This benchmark is therefore a stress test of OPC
*outside* its design regime, not a like-for-like steel reproduction.

Marked ``slow`` and skipped when the (gitignored) Fe-Co h5 is absent.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

from tests.benchmarks.real_feco.harness import (  # noqa: E402
    FECO_ELEMENTS,
    H5_PATH,
    load_real_feco,
    score,
)

# Instrument FWHM (nm) for line extraction -- matches the steel pipeline.
INSTRUMENT_FWHM_NM = 0.2
# Injected n_e for the Saha correction (real Stark width is below the Avantes
# instrument FWHM, exactly as in the steel/ded harnesses).
NE_CM3 = 1e17


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


# --- baseline (plain calibration-free; no OPC) -------------------------------


def solve_baseline(db, wl, inten, truth) -> Dict[str, float]:
    """Plain calibration-free solve: neutral-anchor lines, self-recovered T, no F.

    Uses no standards and no known-matrix calibration, so each sample's result is
    inherently held-out. This is the honest "no-OPC" reference the OPC reduction
    is measured against.
    """
    from tests.benchmarks.ded_precision.line_extractor import extract_line_intensities
    from tests.benchmarks.ded_precision.solver_runner import recovered_wt, run_constrained_solver

    specs = _neutral_anchor_specs(db, wl, list(truth))
    if not specs:
        return {}
    obs = extract_line_intensities(wl, inten, specs, instrument_fwhm_nm=INSTRUMENT_FWHM_NM)
    if not obs:
        return {}
    res = run_constrained_solver(db, obs, NE_CM3, fixed_temperature_K=None)
    return recovered_wt(res)


# --- shipped OPC known-matrix mode (mirrors real_steel/test_opc_shipped) ------


def _build_standards(db, *, thin_filter: bool = False):
    """Build a shipped :class:`Standard` per Fe-Co ladder point.

    Each standard's ``recover(T)`` runs the shipped neutral-anchor selection +
    extraction + constrained Saha-Boltzmann solver at fixed ``T`` (memoized), with
    its certified renormalized truth. Returns ``(standards, samples)``.
    """
    from cflibs.inversion.physics.opc import Standard, StandardRecovery, select_optically_thin_lines
    from tests.benchmarks.ded_precision.line_extractor import extract_line_intensities
    from tests.benchmarks.ded_precision.solver_runner import recovered_wt, run_constrained_solver

    samples = list(load_real_feco())
    standards = []
    for idx, (sid, wl, inten, truth) in enumerate(samples):
        elements = list(truth)

        def _make(wl=wl, inten=inten, elements=elements, thin_filter=thin_filter):
            @lru_cache(maxsize=None)
            def _recover(T_K: float) -> StandardRecovery:
                specs = _neutral_anchor_specs(db, wl, elements)
                if not specs:
                    return StandardRecovery(composition={}, converged=False, degenerate=True)
                obs = extract_line_intensities(
                    wl, inten, specs, instrument_fwhm_nm=INSTRUMENT_FWHM_NM
                )
                if thin_filter:
                    obs = select_optically_thin_lines(wl, inten, obs)
                    if not obs:
                        return StandardRecovery(composition={}, converged=False, degenerate=True)
                res = run_constrained_solver(db, obs, NE_CM3, fixed_temperature_K=float(T_K))
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
    db, wl, inten, elements, calibration, *, thin_filter: bool = False
) -> Dict[str, float]:
    """Held-out OPC solve: neutral-anchor lines -> [thin filter] -> apply_opc -> fixed-T solve."""
    from cflibs.inversion.physics.opc import apply_opc, select_optically_thin_lines
    from tests.benchmarks.ded_precision.line_extractor import extract_line_intensities
    from tests.benchmarks.ded_precision.solver_runner import recovered_wt, run_constrained_solver

    specs = _neutral_anchor_specs(db, wl, elements)
    if not specs:
        return {}
    obs = extract_line_intensities(wl, inten, specs, instrument_fwhm_nm=INSTRUMENT_FWHM_NM)
    if thin_filter:
        obs = select_optically_thin_lines(wl, inten, obs)
        if not obs:
            return {}
    apply_opc(obs, calibration)
    res = run_constrained_solver(db, obs, NE_CM3, fixed_temperature_K=float(calibration.robust_T_K))
    return recovered_wt(res)


def run_baseline_gate(db_path: str = "ASD_da/libs_production.db") -> Dict[str, object]:
    """Plain calibration-free baseline (no OPC) over the Fe-Co ladder."""
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    results: List = []
    for i, (sid, wl, inten, truth) in enumerate(load_real_feco()):
        pred = solve_baseline(db, wl, inten, truth)
        results.append((truth, pred or {}))
        print(f"  [baseline {i + 1}] sample {sid} done", flush=True)
    return {"held_out": score(results)}


def run_shipped_opc_gate(
    db_path: str = "ASD_da/libs_production.db", *, thin_filter: bool = False
) -> Dict[str, object]:
    """Reproduce the held-out RMSEP through the shipped OPC API on Fe-Co (honest LOO)."""
    import warnings

    warnings.filterwarnings("ignore")
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.physics.opc import calibrate_opc

    db = AtomicDatabase(db_path)
    standards, samples = _build_standards(db, thin_filter=thin_filter)

    cal_full = calibrate_opc(standards)
    selected = set(cal_full.selected_standards)
    print(
        f"[opc-feco] selected {len(selected)}/{len(standards)} standards, "
        f"robust_T={cal_full.robust_T_K:.0f}K, F={dict(cal_full.F)}",
        flush=True,
    )

    loo_cache: Dict[str, object] = {}
    results: List = []
    for i, (std, (sid, wl, inten, truth)) in enumerate(zip(standards, samples)):
        name = std.name
        if name in selected:
            if name not in loo_cache:
                loo_cache[name] = calibrate_opc([s for s in standards if s.name != name])
            cal = loo_cache[name]
        else:
            cal = cal_full
        pred = _solve_heldout(db, wl, inten, list(truth), cal, thin_filter=thin_filter)
        results.append((truth, pred or {}))
        print(f"  [opc {i + 1}] sample {sid} done", flush=True)

    return {
        "robust_T_K": cal_full.robust_T_K,
        "n_selected": len(selected),
        "selected_standards": list(cal_full.selected_standards),
        "conditioning_rule": cal_full.conditioning_rule,
        "F": dict(cal_full.F),
        "thin_filter": thin_filter,
        "held_out": score(results),
    }


# --- pytest gate (skips when the gitignored Fe-Co h5 is absent) ---------------


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(H5_PATH), reason=f"Fe-Co h5 not present: {H5_PATH}")
def test_feco_opc_runs_and_is_finite():
    """Smoke gate: the shipped OPC path runs end-to-end on Fe-Co and is finite.

    No RMSEP guard is asserted -- this is an exploratory cross-matrix generality
    benchmark, not a promotion gate (the steel guards live in
    ``real_steel/test_opc_shipped.py``). We only assert the shipped code path is
    exercised on the new alloy system without crashing and yields a finite score.
    """
    out = run_shipped_opc_gate()
    overall = out["held_out"]["rmsep_overall"]
    assert np.isfinite(overall), f"Fe-Co OPC held-out RMSEP not finite: {overall!r}"


def _fmt(d: Dict[str, object]) -> str:
    hs = d["held_out"]
    return ", ".join(f"{k}={hs[k]:.3f}" for k in sorted(hs) if k.startswith("rmsep_"))


if __name__ == "__main__":
    print(f"Fe-Co cross-matrix benchmark, elements={FECO_ELEMENTS}\n")

    base = run_baseline_gate()
    print(f"\n[baseline] held-out RMSEP (wt%): {_fmt(base)}  (n={base['held_out']['n_samples']})\n")

    opc = run_shipped_opc_gate(thin_filter=False)
    print(
        f"\n[opc plain] n_selected={opc['n_selected']} -> {opc['selected_standards']}\n"
        f"           robust_T_K={opc['robust_T_K']:.0f}  F={opc['F']}\n"
        f"           held-out RMSEP (wt%): {_fmt(opc)}\n"
    )

    opc_thin = run_shipped_opc_gate(thin_filter=True)
    print(
        f"\n[opc thin_filter] n_selected={opc_thin['n_selected']} -> "
        f"{opc_thin['selected_standards']}\n"
        f"           robust_T_K={opc_thin['robust_T_K']:.0f}  F={opc_thin['F']}\n"
        f"           held-out RMSEP (wt%): {_fmt(opc_thin)}\n"
    )

    bo = base["held_out"]["rmsep_overall"]
    po = opc["held_out"]["rmsep_overall"]
    to = opc_thin["held_out"]["rmsep_overall"]
    print("=" * 64)
    print(f"baseline {bo:.3f} -> OPC plain {po:.3f} -> OPC thin {to:.3f} wt% (overall held-out)")
    print(f"OPC plain reduction: {bo - po:+.3f} wt% ({(bo - po) / bo * 100:+.1f}%)")
    print(f"OPC thin  reduction: {bo - to:+.3f} wt% ({(bo - to) / bo * 100:+.1f}%)")
    print("=" * 64)
