"""Shipped-API reproduction gate for the constrained known-feedstock DED path.

The DED real goal is *drift tracking on a known matrix*. This is the CRITICAL
validation that the shipped, physics-only constrained force-extraction
(:func:`cflibs.inversion.physics.constrained_extraction.constrained_extract`)
+ matrix-isolation + known-matrix OPC, wired into ``run_pipeline`` as the opt-in
``constrained_extraction`` mode, reproduces the validated real-Mars Ti-6Al-4V
result end-to-end through the SHIPPED pipeline (not the benchmark scripts).

Data: the real SuperCam SCCT_TITANIUM observations under
``data/supercam_calib/raw/scct/cl1/sol_*/`` -- the rover-deck Ti6Al4V
wavelength-calibration plate (Manrique et al. 2020 sect. 2.6) shot repeatedly
through the mission. One point-33 raster per sol gives distinct-sol repeated
measurements of the SAME alloy: the DED known-feedstock scenario (OPC learns the
recovery bias on the standards and is applied held-out to other observations of
the same feedstock). Truth is the nominal CCCT9 Ti6Al4V panel constrained to its
{Ti, Al, V} majors (``cflibs.pds.corpus._CCCT_COMPOSITIONS['CCCT9']`` x100).

Generic peak detection drops Al in ~44% of these dense-Ti-forest spectra even
though Al is well above SNR; the constrained force-extraction measures every
known {Ti, Al, V} line every spectrum, taking the held-out OPC RMSEP from
4.88 wt% (generic detection) to ~0.65 wt% per-shot / ~0.35 averaged. This gate
asserts the SHIPPED path clears the 0.8 wt% guard.

Honesty (structural, by construction, mirrors ``test_opc_shipped.py``):

* The OPC calibration is built by the shipped :func:`calibrate_opc`, which sees
  ONLY standards (their own constrained observations + the certified Ti6Al4V
  panel). It cannot peek at the sample being scored.
* Leave-one-out: a sample that is itself a selected standard is scored with the
  per-element ``F`` re-derived from the OTHER standards only (the established
  honest pattern -- the robust temperature is a ~N-sample average, ``F`` is the
  dominant per-element lever and IS held out).
* The held-out solve runs through the shipped ``run_pipeline`` with
  ``constrained_extraction=True`` and the held-out OPC calibration:
  :func:`apply_opc` rescales only intensities; it never reads a recovered
  composition (no positive-feedback loop).

Marked ``slow`` + ``requires_db``; skipped when the SCCT data or the DB are
absent.
"""

from __future__ import annotations

import glob
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.requires_db]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DB_PATH = _REPO_ROOT / "ASD_da" / "libs_production.db"
_SCCT_GLOB = str(_REPO_ROOT / "data/supercam_calib/raw/scct/cl1/sol_*/*scct_titanium*_33p01.fits")

#: Nominal CCCT9 Ti6Al4V panel x100, constrained to the {Ti, Al, V} majors (the
#: constrained known feedstock set; Fe 0.4% trace excluded, as in the validated
#: lever). Matches scripts/benchmark_pds_opc.py Part 3.
_TI6AL4V_NOMINAL_WT = {"Ti": 89.5, "Al": 6.1, "V": 4.0}
_ELEMENTS = ("Ti", "Al", "V")
#: Per-element forced line budget (Al has only its 4 strong lines outside the
#: SuperCam detector gaps); matches the validated lever.
_BUDGET = {"Ti": 12, "Al": 4, "V": 6}
_RESOLVING_POWER = 2000.0
#: Guard just above the validated per-shot held-out RMSEP (~0.65 wt%); the
#: shipped constrained+OPC path must clear it (vs 4.88 generic detection).
RMSEP_GUARD = 0.8


def _have_inputs() -> bool:
    return _DB_PATH.exists() and bool(glob.glob(_SCCT_GLOB))


def _load_titanium_fits(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Mean spectrum from a SuperCam CL1 SCCT_TITANIUM FITS product."""
    from astropy.io import fits

    from cflibs.benchmark.datasets._common import enforce_strictly_increasing

    with fits.open(path) as hdul:
        wl = np.asarray(hdul["WAVELENGTH"].data["Wavelength"], dtype=float)
        spec_hdu = hdul["SPECTRA"]
        cols = [c for c in spec_hdu.columns.names if c.startswith("Spectrum")]
        stack = np.vstack([np.asarray(spec_hdu.data[c], dtype=float) for c in cols])
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    inten = np.where(finite, stack, 0.0).sum(axis=0) / np.maximum(counts, 1)
    return enforce_strictly_increasing(wl, np.clip(inten, 0.0, None))


def _renorm100(comp: Dict[str, float]) -> Dict[str, float]:
    vals = {e: float(v) for e, v in comp.items() if np.isfinite(float(v)) and float(v) > 0.0}
    tot = sum(vals.values())
    return {e: v / tot * 100.0 for e, v in vals.items()} if tot > 0 else {}


def _score(pairs: List[Tuple[Dict[str, float], Dict[str, float]]]) -> Dict[str, float]:
    """Per-element + overall RMSEP (wt%), truth+pred renormalized to the truth set."""
    per_el: Dict[str, List[float]] = {}
    overall: List[float] = []
    for truth, pred in pairs:
        tn = _renorm100(truth)
        pn = _renorm100({e: pred.get(e, 0.0) for e in tn})
        for e, tv in tn.items():
            err = pn.get(e, 0.0) - tv
            if np.isfinite(err):
                per_el.setdefault(e, []).append(err)
                overall.append(err)
    out = {f"rmsep_{e}": float(np.sqrt(np.mean(np.square(v)))) for e, v in per_el.items()}
    out["rmsep_overall"] = float(np.sqrt(np.mean(np.square(overall)))) if overall else float("nan")
    return out


def _pipeline_overrides(extra: Optional[dict] = None) -> dict:
    """Shipped constrained-extraction pipeline overrides (the validated knobs)."""
    from cflibs.inversion.physics.constrained_extraction import SUPERCAM_DETECTOR_GAPS

    ov = {
        "constrained_extraction": True,
        "constrained_line_budget": dict(_BUDGET),
        "constrained_detector_gaps": list(SUPERCAM_DETECTOR_GAPS),
        "constrained_window_nm": (250.0, 500.0),
        "constrained_instrument_fwhm_nm": 0.18,
        "constrained_search_tol_nm": 0.35,
        "constrained_select_temperature_K": 10000.0,
    }
    if extra:
        ov.update(extra)
    return ov


def _extract_once(db, wl, inten):
    """Shipped constrained extraction (peak-locked, detector-gap dropped) -> obs."""
    from cflibs.inversion.physics.constrained_extraction import (
        SUPERCAM_DETECTOR_GAPS,
        constrained_extract,
    )

    return constrained_extract(
        wl,
        inten,
        db,
        list(_ELEMENTS),
        window=(250.0, 500.0),
        budget=dict(_BUDGET),
        detector_gaps=SUPERCAM_DETECTOR_GAPS,
        instrument_fwhm_nm=0.18,
        search_tol_nm=0.35,
    )


def _solve_wt(db, observations, fixed_T: Optional[float]) -> Dict[str, float]:
    """Recover wt% from cached constrained obs at fixed T (shipped iterative solver)."""
    from cflibs.inversion.pipeline import _number_to_mass_fractions
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

    if not observations:
        return {}
    solver = IterativeCFLIBSSolver(
        atomic_db=db,
        saha_boltzmann_graph=True,
        apply_self_absorption="off",
        assess_quality=False,
        fixed_temperature_K=fixed_T,
    )
    res = solver.solve(list(observations), closure_mode="standard")
    mass = res.mass_fractions or _number_to_mass_fractions(res.concentrations)
    return {k: 100.0 * v for k, v in mass.items()}


def _build_standard(db, name, observations, truth_wt):
    """Shipped Standard whose recover(T) solves cached constrained obs at fixed T."""
    from cflibs.inversion.physics.opc import Standard, StandardRecovery

    @lru_cache(maxsize=None)
    def _recover(T_K: float) -> StandardRecovery:
        pred = _solve_wt(db, observations, float(T_K))
        if not pred:
            return StandardRecovery(composition={}, converged=False, degenerate=True)
        return StandardRecovery(composition=pred, converged=True, degenerate=False)

    return Standard(name=name, certified=dict(truth_wt), recover=_recover)


def run_shipped_constrained_titanium_gate(
    db_path: str = str(_DB_PATH), limit: Optional[int] = None
) -> Dict[str, object]:
    """Held-out constrained+OPC RMSEP through the SHIPPED pipeline (honest LOO).

    1. Shipped constrained extraction of every Ti6Al4V SCCT spectrum (once each).
    2. ``calibrate_opc`` over the per-spectrum standards (in-sample conditioning
       gate + robust ``T`` = mean optimal-``T*`` of the selected standards).
    3. Per-selected-standard ``F`` at robust ``T``; robust + leave-one-out
       geometric-mean ``F`` (clamp-saturated filtered).
    4. Held-out solve per spectrum via ``run_pipeline`` with
       ``constrained_extraction=True`` and the held-out OPC calibration.
    """
    import warnings

    warnings.filterwarnings("ignore")

    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.pipeline import build_pipeline_config, run_pipeline
    from cflibs.inversion.physics.opc import (
        OPCCalibration,
        _derive_F,
        _geomean_F,
        calibrate_opc,
    )

    db = AtomicDatabase(db_path)
    paths = sorted(glob.glob(_SCCT_GLOB))
    if limit:
        paths = paths[:limit]

    # 1. Extract every spectrum once via the shipped constrained extraction.
    samples = []  # (label, observations, truth)
    for p in paths:
        sol = Path(p).name.split("_")[1]
        wl, inten = _load_titanium_fits(p)
        obs = _extract_once(db, wl, inten)
        if obs:
            samples.append((f"sol{sol}", obs, dict(_TI6AL4V_NOMINAL_WT), wl, inten))

    al_rate = (
        float(np.mean([any(o.element == "Al" for o in s[1]) for s in samples])) if samples else 0.0
    )

    # 2. Calibrate OPC over the per-spectrum standards (shipped, standards only).
    standards = [_build_standard(db, lbl, obs, truth) for (lbl, obs, truth, _, _) in samples]
    cal = calibrate_opc(standards)
    selected = list(cal.selected_standards)
    selected_set = set(selected)
    by_name = {s.name: s for s in standards}
    robust_T = float(cal.robust_T_K)

    # 3. Per-selected-standard F at robust_T; robust + leave-one-out geomean F.
    F_per = {n: _derive_F(by_name[n], robust_T) for n in selected}
    robust_F = _geomean_F(list(F_per.values()))
    loo_F = {n: (_geomean_F([F_per[m] for m in selected if m != n]) or robust_F) for n in selected}

    # 4. Held-out solve per spectrum through the SHIPPED run_pipeline (constrained
    #    extraction + the held-out OPC calibration).
    baseline_pairs: List[Tuple[Dict, Dict]] = []
    opc_pairs: List[Tuple[Dict, Dict]] = []
    for label, obs, truth, wl, inten in samples:
        baseline_pairs.append((truth, _solve_wt(db, obs, None)))

        F = loo_F[label] if label in selected_set else robust_F
        fold_cal = OPCCalibration(
            robust_T_K=robust_T,
            F=dict(F),
            selected_standards=selected,
            conditioning_rule=cal.conditioning_rule,
        )
        cfg = build_pipeline_config(
            list(_ELEMENTS),
            preset="metallic_ded",
            resolving_power=_RESOLVING_POWER,
            overrides=_pipeline_overrides({"opc": fold_cal}),
        )
        try:
            res, _diag = run_pipeline(wl, inten, db, cfg)
            pred = {k: 100.0 * v for k, v in (res.mass_fractions or {}).items()}
        except Exception:  # noqa: BLE001 — a failed solve scores as empty, not a crash
            pred = {}
        opc_pairs.append((truth, pred))

    return {
        "n_observations": len(samples),
        "al_measured_rate": al_rate,
        "robust_T_K": robust_T,
        "n_selected": len(selected),
        "robust_F": {str(k): float(v) for k, v in robust_F.items()},
        "baseline": _score(baseline_pairs),
        "opc_heldout": _score(opc_pairs),
    }


@pytest.mark.skipif(not _have_inputs(), reason="SCCT Ti6Al4V data or libs_production.db absent")
def test_shipped_constrained_titanium_opc_heldout_below_guard():
    out = run_shipped_constrained_titanium_gate()
    assert out["n_observations"] >= 8, f"too few Ti6Al4V spectra: {out['n_observations']}"
    # Constrained extraction must measure Al on every spectrum (the whole point).
    assert out["al_measured_rate"] == pytest.approx(1.0), (
        f"constrained extraction failed to measure Al on every spectrum "
        f"(rate={out['al_measured_rate']:.2f}); generic detection drops it ~44% of the time."
    )
    held = out["opc_heldout"]
    overall = held["rmsep_overall"]
    assert np.isfinite(overall), f"shipped constrained+OPC held-out RMSEP not finite: {overall!r}"
    assert overall <= RMSEP_GUARD, (
        f"shipped constrained+OPC Ti6Al4V reproduction regressed: held-out rmsep_overall "
        f"{overall:.3f} > guard {RMSEP_GUARD} (validated ~0.65 per-shot, vs 4.88 generic). "
        "per-element: "
        + ", ".join(f"{k}={held[k]:.2f}" for k in sorted(held) if k != "rmsep_overall")
    )


if __name__ == "__main__":
    res = run_shipped_constrained_titanium_gate()
    print("\n[shipped-constrained:titanium] real Mars Ti6Al4V held-out RMSEP (wt%):")
    print(f"  n_observations:   {res['n_observations']}")
    print(f"  Al-measured rate: {res['al_measured_rate']:.2f}")
    print(f"  robust_T_K:       {res['robust_T_K']:.0f}")
    print(f"  n_selected:       {res['n_selected']}")
    print(f"  robust_F:         {res['robust_F']}")
    print(f"  baseline:    {res['baseline']}")
    print(f"  OPC held-out:{res['opc_heldout']}")
