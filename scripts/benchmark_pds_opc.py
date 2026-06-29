#!/usr/bin/env python3
"""Real-data composition benchmark on ChemCam/SuperCam calibration spectra (OPC).

Validates the opt-in known-matrix OPC mode (``cflibs.inversion.physics.opc``)
on REAL planetary-LIBS calibration data, the canonical CF-LIBS testbed:

* ``--bhvo2``  : the 12 in-repo BHVO-2 (USGS basalt geostandard) spectra from
  three instruments (ChemCam / SuperCam / CSA). Reports calibration-free RMSEP
  per instrument source + a same-composition leave-one-spectrum-out OPC demo
  (a single composition cannot test cross-composition generalization -- that is
  the ``--chemcam`` benchmark; this only holds out the noise realization).

* ``--chemcam``: the 60 usable preflight cleanroom standards (Wiens 2013) from
  the PDS MSL-M-CHEMCAM-LIBS-4/5-RDR-V1.0 CALIB directory
  (``data/chemcam_calib/``; 4 replicates per standard, averaged). Reports the
  calibration-free baseline AND an HONEST held-out OPC RMSEP: ``calibrate_opc``
  sees only the standards' own spectra+truth and a selected standard is scored
  with a leave-one-out geomean ``F`` re-derived from the OTHER standards (the
  same honest-LOO pattern as ``tests/benchmarks/real_steel/test_opc_shipped``).
  This is a real, 60-distinct-composition held-out composition benchmark.

Everything routes through the SHIPPED physics-only code: the production
detection+selection (``detect_and_select_lines``, geological preset knobs), the
shipped Stark n_e diagnostic, the shipped ``IterativeCFLIBSSolver`` (oxide
closure), and the shipped ``calibrate_opc`` / ``apply_opc``. Detection (the
expensive RANSAC wavelength calibration) is run ONCE per spectrum and the
cached observations are re-solved at each fixed temperature, so the optimal-T
scan + held-out LOO are cheap.

Usage::

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/benchmark_pds_opc.py --bhvo2
    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/benchmark_pds_opc.py --chemcam
    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/benchmark_pds_opc.py --chemcam --limit 8
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings.filterwarnings("ignore")


def _db_path() -> str:
    for cand in (
        _REPO_ROOT / "ASD_da/libs_production.db",
        Path("ASD_da/libs_production.db"),
    ):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError("libs_production.db not found")


# --------------------------------------------------------------------------- #
# Shared detect-once / solve-many helpers (the geological production path)     #
# --------------------------------------------------------------------------- #


def detect_observations(
    db,
    wl: np.ndarray,
    inten: np.ndarray,
    elements: Sequence[str],
    resolving_power: Optional[float],
    preset: str = "geological",
    matrix_isolation_element: Optional[str] = None,
):
    """Run the shipped detection + Stark n_e ONCE for the given preset.

    Returns ``(observations, stark_diagnostics)``. Mirrors exactly the
    detection/selection knobs ``run_pipeline`` resolves from ``preset``
    (``geological`` oxide rocks; ``metallic_ded`` for a constrained alloy set),
    so the cached observations are the ones the production pipeline would solve.
    The expensive RANSAC wavelength calibration runs here once.
    """
    from cflibs.inversion.pipeline import build_pipeline_config, detect_and_select_lines
    from cflibs.inversion.physics.stark_ne import measure_stark_ne

    overrides = (
        {"matrix_isolation_element": matrix_isolation_element}
        if matrix_isolation_element is not None
        else None
    )
    pipe = build_pipeline_config(
        list(elements), preset=preset, resolving_power=resolving_power, overrides=overrides
    )
    obs = detect_and_select_lines(
        wl,
        inten,
        db,
        pipe.elements,
        min_relative_intensity=pipe.min_relative_intensity,
        top_k_per_element=pipe.top_k_per_element,
        resolving_power=pipe.resolving_power,
        wavelength_tolerance_nm=pipe.wavelength_tolerance_nm,
        min_peak_height=pipe.min_peak_height,
        peak_width_nm=pipe.peak_width_nm,
        apply_self_absorption=pipe.apply_self_absorption,
        exclude_resonance=pipe.exclude_resonance,
        min_snr=pipe.min_snr,
        min_energy_spread_ev=pipe.min_energy_spread_ev,
        min_lines_per_element=pipe.min_lines_per_element,
        isolation_wavelength_nm=pipe.isolation_wavelength_nm,
        max_lines_per_element=pipe.max_lines_per_element,
        wavelength_calibration=pipe.wavelength_calibration,
        shift_coherence_veto=pipe.shift_coherence_veto,
        residual_shift_scan_nm=pipe.residual_shift_scan_nm,
        global_shift_scan_nm=pipe.global_shift_scan_nm,
        affine_coverage_gate=pipe.affine_coverage_gate,
        line_residual_gate=pipe.line_residual_gate,
        matrix_isolation_element=pipe.matrix_isolation_element,
        matrix_isolation_n_fwhm=pipe.matrix_isolation_n_fwhm,
        matrix_isolation_contamination_ratio=pipe.matrix_isolation_contamination_ratio,
        return_diagnostics=False,
    )
    stark_diag = None
    if obs:
        try:
            sr = measure_stark_ne(wl, inten, obs, db, resolving_power=resolving_power)
            if sr is not None and sr.usable:
                stark_diag = sr.diagnostics
        except Exception:  # noqa: BLE001
            stark_diag = None
    return obs, stark_diag


def _solver(db, fixed_T: Optional[float]):
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

    return IterativeCFLIBSSolver(
        atomic_db=db,
        saha_boltzmann_graph=True,
        apply_self_absorption="off",
        assess_quality=False,  # composition byte-identical; skips the perf re-fit
        fixed_temperature_K=fixed_T,
    )


def solve_mass_wt(
    db, observations, stark_diag, fixed_T: Optional[float], closure_mode: str = "oxide"
) -> Dict[str, float]:
    """Solve cached observations -> recovered composition in wt%.

    ``closure_mode`` ``oxide`` for geological rocks; ``standard`` (sum-to-one,
    no oxide stoichiometry) for a metal alloy. ``fixed_T`` None recovers T from
    the Boltzmann slope (calibration-free baseline); a value holds the plasma
    temperature (OPC / optimal-T).
    """
    from cflibs.inversion.physics.closure import default_oxide_stoichiometry
    from cflibs.inversion.pipeline import _number_to_mass_fractions

    if not observations:
        return {}
    els = [o.element for o in observations]
    solver = _solver(db, fixed_T)
    closure_kwargs = (
        {"oxide_stoichiometry": default_oxide_stoichiometry(els)} if closure_mode == "oxide" else {}
    )
    res = solver.solve(
        list(observations),
        closure_mode=closure_mode,
        stark_diagnostics=stark_diag,
        **closure_kwargs,
    )
    mass = res.mass_fractions or _number_to_mass_fractions(res.concentrations)
    degenerate = float((res.quality_metrics or {}).get("degenerate_composition", 0.0)) >= 0.5
    out = {k: 100.0 * v for k, v in mass.items()}
    out["__converged__"] = float(bool(res.converged))
    out["__degenerate__"] = float(degenerate)
    return out


def _strip(pred: Dict[str, float]) -> Dict[str, float]:
    return {k: v for k, v in pred.items() if not k.startswith("__")}


# --------------------------------------------------------------------------- #
# Scoring                                                                       #
# --------------------------------------------------------------------------- #


def renorm100(comp: Dict[str, float]) -> Dict[str, float]:
    vals = {e: float(v) for e, v in comp.items() if np.isfinite(float(v)) and float(v) > 0}
    tot = sum(vals.values())
    return {e: v / tot * 100.0 for e, v in vals.items()} if tot > 0 else {}


def score(pairs: List[Tuple[Dict[str, float], Dict[str, float]]]) -> Dict[str, float]:
    """Per-element + overall RMSEP (wt%). Truth AND pred renormalized to 100%
    over the truth's element set (closed-basis comparison)."""
    per_el: Dict[str, List[float]] = {}
    overall: List[float] = []
    for truth, pred in pairs:
        tn = renorm100(truth)
        pn = renorm100({e: _strip(pred).get(e, 0.0) for e in tn})
        for e, tv in tn.items():
            err = pn.get(e, 0.0) - tv
            if np.isfinite(err):
                per_el.setdefault(e, []).append(err)
                overall.append(err)
    out = {f"rmsep_{e}": float(np.sqrt(np.mean(np.square(v)))) for e, v in per_el.items()}
    out["rmsep_overall"] = float(np.sqrt(np.mean(np.square(overall)))) if overall else float("nan")
    out["n_pairs"] = len(pairs)
    return out


def _fmt_score(s: Dict[str, float]) -> str:
    parts = [f"overall={s['rmsep_overall']:.3f}"]
    for k in sorted(s):
        if k.startswith("rmsep_") and k != "rmsep_overall":
            parts.append(f"{k[6:]}={s[k]:.2f}")
    return "  ".join(parts)


# --------------------------------------------------------------------------- #
# OPC helpers (shipped calibrate_opc + honest held-out LOO)                    #
# --------------------------------------------------------------------------- #


def build_standard(db, name, observations, stark_diag, truth_wt, closure_mode: str = "oxide"):
    """A shipped ``Standard`` whose recover(T) solves cached obs at fixed T."""
    from functools import lru_cache

    from cflibs.inversion.physics.opc import Standard, StandardRecovery

    @lru_cache(maxsize=None)
    def _recover(T_K: float) -> StandardRecovery:
        pred = solve_mass_wt(db, observations, stark_diag, float(T_K), closure_mode)
        if not _strip(pred):
            return StandardRecovery(composition={}, converged=False, degenerate=True)
        return StandardRecovery(
            composition=_strip(pred),
            converged=pred.get("__converged__", 0.0) >= 0.5,
            degenerate=pred.get("__degenerate__", 0.0) >= 0.5,
        )

    return Standard(name=name, certified=dict(truth_wt), recover=_recover)


def opc_heldout(db, samples, closure_mode: str = "oxide"):
    """Honest held-out OPC over ``samples`` = [(name, obs, stark, truth_wt), ...].

    Returns ``(baseline_pairs, opc_pairs, provenance)``. The OPC ``F`` is
    re-derived leave-one-out (a selected standard never sees its own ``F``);
    the robust temperature is held at the full-calibration value across folds
    (the established honest pattern in test_opc_shipped -- T is a ~N-sample
    average, F is the dominant per-element lever and IS held out).
    """
    from cflibs.inversion.physics.opc import (
        OPCCalibration,
        _derive_F,
        _geomean_F,
        calibrate_opc,
    )

    standards = [build_standard(db, n, o, s, t, closure_mode) for (n, o, s, t) in samples]
    cal = calibrate_opc(standards)
    selected = list(cal.selected_standards)
    selected_set = set(selected)
    by_name = {st.name: st for st in standards}
    robust_T = float(cal.robust_T_K)

    # Per-selected-standard F at the full-calibration robust_T (cached recover).
    F_per = {n: _derive_F(by_name[n], robust_T) for n in selected}
    robust_F = _geomean_F(list(F_per.values()))
    loo_F = {n: (_geomean_F([F_per[m] for m in selected if m != n]) or robust_F) for n in selected}

    baseline_pairs: List[Tuple[Dict, Dict]] = []
    opc_pairs: List[Tuple[Dict, Dict]] = []
    from cflibs.inversion.physics.opc import apply_opc

    for name, obs, stark, truth in samples:
        base = solve_mass_wt(db, obs, stark, None, closure_mode)
        baseline_pairs.append((truth, base))

        F = loo_F[name] if name in selected_set else robust_F
        fold_cal = OPCCalibration(
            robust_T_K=robust_T,
            F=F,
            selected_standards=selected,
            conditioning_rule=cal.conditioning_rule,
        )
        obs_opc = copy.deepcopy(list(obs))
        apply_opc(obs_opc, fold_cal)
        pred = solve_mass_wt(db, obs_opc, stark, robust_T, closure_mode)
        opc_pairs.append((truth, pred))

    prov = {
        "robust_T_K": robust_T,
        "n_selected": len(selected),
        "selected_standards": selected,
        "conditioning_rule": cal.conditioning_rule,
        "robust_F": {str(k): float(v) for k, v in robust_F.items()},
    }
    return baseline_pairs, opc_pairs, prov


# --------------------------------------------------------------------------- #
# Part 1 -- BHVO-2 (in-repo, 12 spectra, 4 instrument sources)                 #
# --------------------------------------------------------------------------- #

_BHVO2_SOURCES = {
    "chemcam": [
        "chemcam_bhvo2_loc1_spectrum.csv",
        "chemcam_bhvo2_loc2_spectrum.csv",
        "chemcam_bhvo2_loc3_spectrum.csv",
        "chemcam_bhvo2_loc4_spectrum.csv",
    ],
    "supercam_1545mm": [
        "supercam_1545mm_loc1_spectrum.csv",
        "supercam_1545mm_loc2_spectrum.csv",
        "supercam_1545mm_loc3_spectrum.csv",
    ],
    "supercam_4250mm": [
        "supercam_4250mm_loc1_spectrum.csv",
        "supercam_4250mm_loc2_spectrum.csv",
        "supercam_4250mm_loc3_spectrum.csv",
    ],
    "csa": ["csa_bhvo2_1000pulse_spectrum.csv", "csa_bhvo2_200pulse_spectrum.csv"],
}
_BHVO2_RP = {"chemcam": 2000.0, "supercam_1545mm": 2000.0, "supercam_4250mm": 2000.0, "csa": None}


def run_bhvo2(db) -> dict:
    from cflibs.benchmark.reference_compositions import BHVO2_BASALT_USGS
    from cflibs.io.spectrum import load_spectrum

    data_dir = _REPO_ROOT / "data" / "bhvo2_usgs"
    elements = list(BHVO2_BASALT_USGS)
    truth_wt = {e: 100.0 * v for e, v in BHVO2_BASALT_USGS.items()}

    print("\n" + "=" * 74)
    print("  PART 1 -- BHVO-2 (USGS basalt) real-data composition accuracy")
    print("=" * 74)

    out: dict = {"certified_wt": truth_wt, "by_source": {}}
    all_base: List[Tuple[Dict, Dict]] = []
    all_opc: List[Tuple[Dict, Dict]] = []
    for src, files in _BHVO2_SOURCES.items():
        rp = _BHVO2_RP[src]
        samples = []
        for fn in files:
            wl, inten = load_spectrum(str(data_dir / fn))
            inten = np.clip(np.asarray(inten, dtype=float), 0.0, None)
            t0 = time.perf_counter()
            obs, stark = detect_observations(db, wl, inten, elements, rp)
            samples.append((fn, obs, stark, truth_wt))
            print(f"  [{src}] {fn}: {len(obs)} obs, detect {time.perf_counter()-t0:.1f}s")
        # baseline per spectrum
        base_pairs = [(truth_wt, solve_mass_wt(db, o, s, None)) for (_, o, s, _) in samples]
        bsc = score(base_pairs)
        all_base.extend(base_pairs)
        src_rec: dict = {"n_spectra": len(samples), "baseline": bsc}
        # same-composition leave-one-spectrum-out OPC (noise hold-out only) when >=2 reps
        if len(samples) >= 2:
            bp, op, prov = opc_heldout(db, samples)
            osc = score(op)
            all_opc.extend(op)
            src_rec["opc_loo"] = osc
            src_rec["opc_prov"] = prov
            print(f"  [{src}] baseline {_fmt_score(bsc)}")
            print(
                f"  [{src}] OPC-LOO  {_fmt_score(osc)}  (selected={prov['n_selected']}, "
                f"robust_T={prov['robust_T_K']:.0f}K)"
            )
        else:
            print(f"  [{src}] baseline {_fmt_score(bsc)}  (OPC LOO needs >=2 reps)")
        out["by_source"][src] = src_rec
    out["overall_baseline"] = score(all_base)
    out["overall_opc_loo"] = score(all_opc) if all_opc else None
    print("-" * 74)
    print(f"  BHVO-2 OVERALL baseline (12 spectra): {_fmt_score(out['overall_baseline'])}")
    if out["overall_opc_loo"]:
        print(f"  BHVO-2 OVERALL OPC same-comp LOO    : {_fmt_score(out['overall_opc_loo'])}")
        print("  NOTE: BHVO-2 is a single composition -> OPC LOO holds out only the noise")
        print("        realization, NOT composition. The honest cross-composition held-out")
        print("        OPC test for ChemCam BHVO-2 is delivered by --chemcam (BHVO2 is one")
        print("        of the 60 standards there).")
    return out


# --------------------------------------------------------------------------- #
# Part 2 -- 60-standard ChemCam preflight held-out OPC benchmark               #
# --------------------------------------------------------------------------- #


def run_chemcam(db, limit: Optional[int]) -> dict:
    from cflibs.benchmark.datasets import chemcam_calib

    root = _REPO_ROOT / "data" / "chemcam_calib"
    print("\n" + "=" * 74)
    print("  PART 2 -- ChemCam preflight standards: held-out OPC composition benchmark")
    print("=" * 74)

    # Average the 4 replicates per standard -> one mean spectrum per composition.
    by_std: Dict[str, dict] = {}
    for sid, wl, inten, truth in chemcam_calib.iter_spectra(root):
        name = sid.split("/")[1]
        if truth.composition_wt is None:
            continue
        rec = by_std.setdefault(name, {"wl": wl, "stack": [], "truth": truth.composition_wt})
        # all replicates share the strictly-increasing axis; guard length
        if inten.shape == rec["wl"].shape:
            rec["stack"].append(inten)
    names = sorted(by_std)
    if limit:
        names = names[:limit]
    print(f"  {len(names)} distinct standards (4 reps averaged each)")

    samples = []
    for i, name in enumerate(names):
        rec = by_std[name]
        wl = rec["wl"]
        inten = np.clip(np.mean(np.vstack(rec["stack"]), axis=0), 0.0, None)
        elements = sorted(rec["truth"])
        t0 = time.perf_counter()
        obs, stark = detect_observations(db, wl, inten, elements, 2000.0)
        samples.append((name, obs, stark, rec["truth"]))
        print(
            f"  [{i+1}/{len(names)}] {name}: {len(obs)} obs, "
            f"detect {time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    print("  building OPC calibration + held-out solves ...", flush=True)
    t0 = time.perf_counter()
    base_pairs, opc_pairs, prov = opc_heldout(db, samples)
    print(f"  OPC + held-out solves done in {time.perf_counter()-t0:.1f}s", flush=True)

    bsc = score(base_pairs)
    osc = score(opc_pairs)
    print("-" * 74)
    print(f"  baseline (calibration-free) : {_fmt_score(bsc)}")
    print(f"  OPC held-out (honest LOO)   : {_fmt_score(osc)}")
    print(
        f"  selected standards: {prov['n_selected']}/{len(samples)}  "
        f"robust_T={prov['robust_T_K']:.0f}K"
    )
    return {
        "n_standards": len(samples),
        "baseline": bsc,
        "opc_heldout": osc,
        "provenance": prov,
        "per_standard": [
            {
                "name": n,
                "truth_wt": renorm100(t),
                "baseline_wt": renorm100(_strip(bp)),
                "opc_wt": renorm100(_strip(op)),
            }
            for (n, _, _, t), (_, bp), (_, op) in zip(samples, base_pairs, opc_pairs)
        ],
    }


# --------------------------------------------------------------------------- #
# Part 3 -- Ti-6Al-4V (DED matrix): real Mars SuperCam SCCT_TITANIUM spectra   #
# --------------------------------------------------------------------------- #

#: Nominal Ti-6Al-4V (grade 5) certified panel, from the on-deck ChemCam CCCT9
#: alloy plate (cflibs.pds.corpus._CCCT_COMPOSITIONS["CCCT9"]); the SuperCam
#: SCCT_TITANIUM plate is the same Ti6Al4V wavelength-calibration alloy
#: (Manrique et al. 2020, sect. 2.6). No per-target EMPA panel exists, so the
#: alloy's nominal constituents are the truth for the constrained {Ti,Al,V} solve.
_TI6AL4V_NOMINAL_WT = {"Ti": 89.5, "Al": 6.1, "V": 4.0}  # corpus CCCT9 x100
_TI6AL4V_ELEMENTS = ("Ti", "Al", "V")

#: SuperCam 3-spectrometer detector gaps (nm): UV|VIO and VIO|VNIR boundaries.
#: Lines falling in a gap are physically unmeasurable, so the forced known-line
#: list drops them (measured on the data: 341.4-379.3 and 464.5-537.6).
_SUPERCAM_DETECTOR_GAPS = ((341.4, 379.3), (464.5, 537.6))
#: Per-element budget for the forced Ti-6Al-4V line list. Al has only its four
#: strong resonance lines (394.4/396.15/308.2/309.27) outside the gaps.
_TI6AL4V_FORCED_BUDGET = {"Ti": 12, "Al": 4, "V": 6}
_FWHM_TO_SIGMA = 1.0 / 2.354820045


def _in_detector_gap(wavelength_nm: float) -> bool:
    return any(lo <= wavelength_nm <= hi for lo, hi in _SUPERCAM_DETECTOR_GAPS)


def build_titanium_line_list(
    db,
    elements: Sequence[str],
    window: Tuple[float, float] = (250.0, 500.0),
    T_K: float = 10000.0,
    budget: Optional[Dict[str, int]] = None,
):
    """Curated KNOWN-position {Ti,Al,V} line list for constrained force-extraction.

    Reuses the DED ``select_lines`` emissivity ranking -- the line positions and
    atomic parameters come from the atomic DB, NOT from the truth composition, so
    this is honest (it encodes only the KNOWN element set, the legitimate DED
    feedstock assumption). Resonance lines are KEPT (Al's only strong lines in
    this window are its 394.4/396.15 and 308.2/309.27 resonance doublets), and
    lines that fall in a SuperCam detector gap are dropped.
    """
    from tests.benchmarks.ded_precision.line_lists import select_lines

    budget = budget or _TI6AL4V_FORCED_BUDGET
    specs = []
    for el in elements:
        n = budget.get(el, 8)
        cand = select_lines(
            db, el, window, n * 4, T_K=T_K, exclude_resonance=False, prefer_spread=False
        )
        cand = [s for s in cand if not _in_detector_gap(s.wavelength_nm)]
        specs.extend(cand[:n])
    return specs


def _extract_peak_locked(wl, inten, specs, instrument_fwhm_nm: float, search_tol_nm: float):
    """Peak-locked windowed extraction at known line positions -> observations.

    For each known LineSpec, lock onto the local peak within +/-``search_tol_nm``
    of the catalog wavelength (handles the SuperCam spectrometer-dependent
    wavelength offset measured on these data: ~+0.15 nm in VIO, ~-0.3 nm in UV),
    then integrate over +/-1.5 sigma after a robust local linear baseline. This
    GUARANTEES every known line is measured -- including the dense-Ti-blended Al
    lines that generic peak detection drops (~44% of spectra here).
    """
    from cflibs.inversion.common import LineObservation

    _trapz = getattr(np, "trapezoid", None) or np.trapz
    wl = np.asarray(wl, dtype=float)
    inten = np.asarray(inten, dtype=float)
    if wl.size < 5:
        return []
    step = float(np.median(np.diff(wl)))
    sigma = instrument_fwhm_nm * _FWHM_TO_SIGMA
    half = max(1.5 * sigma, 3.0 * step)
    obs = []
    for ls in specs:
        sm = (wl >= ls.wavelength_nm - search_tol_nm) & (wl <= ls.wavelength_nm + search_tol_nm)
        if int(sm.sum()) < 5:
            continue
        xs, ys = wl[sm], inten[sm]
        k = max(1, xs.size // 6)
        base = np.interp(xs, [xs[0], xs[-1]], [np.median(ys[:k]), np.median(ys[-k:])])
        peak_wl = float(xs[int(np.argmax(ys - base))])
        im = (wl >= peak_wl - half) & (wl <= peak_wl + half)
        if int(im.sum()) < 4:
            continue
        x, y = wl[im], inten[im].astype(float)
        k2 = max(1, x.size // 6)
        b = np.interp(x, [x[0], x[-1]], [np.median(y[:k2]), np.median(y[-k2:])])
        area = float(_trapz(y - b, x))
        if not np.isfinite(area) or area <= 0.0:
            continue
        obs.append(
            LineObservation(
                wavelength_nm=peak_wl,
                intensity=area,
                intensity_uncertainty=max(area * 0.05, 1e-12),
                element=ls.element,
                ionization_stage=ls.ionization_stage,
                E_k_ev=ls.E_k_ev,
                g_k=ls.g_k,
                A_ki=ls.A_ki,
                aki_uncertainty=ls.aki_uncertainty,
            )
        )
    return obs


def forced_observations(
    db,
    wl: np.ndarray,
    inten: np.ndarray,
    specs,
    instrument_fwhm_nm: float = 0.18,
    search_tol_nm: float = 0.35,
):
    """Force-extract known lines + the shipped Stark n_e diagnostic on the result."""
    from cflibs.inversion.physics.stark_ne import measure_stark_ne

    obs = _extract_peak_locked(wl, inten, specs, instrument_fwhm_nm, search_tol_nm)
    stark = None
    if obs:
        try:
            sr = measure_stark_ne(wl, inten, obs, db, resolving_power=2000.0)
            if sr is not None and sr.usable:
                stark = sr.diagnostics
        except Exception:  # noqa: BLE001
            stark = None
    return obs, stark


def _load_titanium_fits(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Mean spectrum from a SuperCam CL1 SCCT_TITANIUM FITS product."""
    from astropy.io import fits

    with fits.open(path) as hdul:
        wl = np.asarray(hdul["WAVELENGTH"].data["Wavelength"], dtype=float)
        spec_hdu = hdul["SPECTRA"]
        cols = [c for c in spec_hdu.columns.names if c.startswith("Spectrum")]
        stack = np.vstack([np.asarray(spec_hdu.data[c], dtype=float) for c in cols])
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    inten = np.where(finite, stack, 0.0).sum(axis=0) / np.maximum(counts, 1)
    from cflibs.benchmark.datasets._common import enforce_strictly_increasing

    return enforce_strictly_increasing(wl, np.clip(inten, 0.0, None))


def run_titanium(
    db,
    limit: Optional[int],
    matrix_isolation: bool = False,
    forced: bool = False,
    avg_group: int = 1,
) -> dict:
    """Constrained {Ti,Al,V} solve on real Mars SuperCam Ti-6Al-4V spectra.

    The SCCT_TITANIUM target is the rover-deck Ti6Al4V wavelength-calibration
    plate -- a true DED-matrix alloy shot repeatedly through the mission. OPC is
    a same-composition leave-one-observation-out (the DED scenario: a KNOWN
    feedstock matrix, OPC learns the recovery bias and is applied held-out to
    other observations of the same alloy). Truth is the nominal Ti-6Al-4V panel.

    ``forced`` switches the observation source from generic peak detection to
    constrained force-extraction at the KNOWN {Ti,Al,V} line positions
    (:func:`build_titanium_line_list` + :func:`forced_observations`). Generic
    detection drops Al in ~44% of these dense-Ti-forest spectra even though Al
    is well above SNR; force-extraction measures every known line every spectrum.

    ``avg_group`` averages consecutive-sol spectra into higher-SNR composites
    before extraction (the multi-shot averaging SNR lever); ``avg_group=1`` keeps
    one independent observation per sol (the strongest held-out OPC test).
    """
    print("\n" + "=" * 74)
    print("  PART 3 -- Ti-6Al-4V (DED matrix): real Mars SuperCam SCCT_TITANIUM")
    print("=" * 74)
    cl1 = _REPO_ROOT / "data" / "supercam_calib" / "raw" / "scct" / "cl1"
    # One canonical raster (point 33) per sol -> distinct-sol repeated measurements.
    paths = sorted(cl1.glob("sol_*/*scct_titanium*_33p01.fits"))
    if not paths:
        paths = sorted(cl1.glob("sol_*/*scct_titanium*.fits"))
    if limit:
        paths = paths[:limit]
    mode = "FORCED known-position extraction" if forced else "generic detection"
    print(
        f"  {len(paths)} Ti-6Al-4V observations (one point-33 raster per sol); "
        f"nominal truth Ti/Al/V = {_TI6AL4V_NOMINAL_WT}"
    )
    print(f"  mode = {mode};  avg_group = {avg_group}")

    elements = list(_TI6AL4V_ELEMENTS)
    # Dominant matrix element = the most abundant nominal constituent (Ti).
    matrix_el = (
        max(_TI6AL4V_NOMINAL_WT, key=_TI6AL4V_NOMINAL_WT.get)
        if (matrix_isolation and not forced)
        else None
    )
    if matrix_el is not None:
        print(f"  matrix-isolation ON: dropping trace lines blended with {matrix_el}")
    specs = None
    if forced:
        specs = build_titanium_line_list(db, elements)
        scnt = {e: sum(1 for s in specs if s.element == e) for e in elements}
        print(f"  forced line list: {len(specs)} known lines {scnt}")

    # Load all spectra onto the shared instrument wavelength axis, then group
    # consecutive sols into avg_group-sized composites (averaging SNR lever).
    loaded: List[Tuple[str, np.ndarray]] = []
    wl_ref: Optional[np.ndarray] = None
    for p in paths:
        sol = p.name.split("_")[1]
        wl, inten = _load_titanium_fits(p)
        if wl_ref is None:
            wl_ref = wl
        if inten.shape != wl_ref.shape:
            continue
        loaded.append((sol, inten))

    samples = []
    n_groups = (len(loaded) + avg_group - 1) // avg_group
    for gi in range(n_groups):
        chunk = loaded[gi * avg_group : (gi + 1) * avg_group]
        if not chunk:
            continue
        sols = [c[0] for c in chunk]
        label = f"sol{sols[0]}" if len(sols) == 1 else f"sol{sols[0]}-{sols[-1]}"
        inten = chunk[0][1] if len(chunk) == 1 else np.mean(np.vstack([c[1] for c in chunk]), 0)
        t0 = time.perf_counter()
        if forced:
            obs, stark = forced_observations(db, wl_ref, inten, specs)
        else:
            obs, stark = detect_observations(
                db,
                wl_ref,
                inten,
                elements,
                2000.0,
                preset="metallic_ded",
                matrix_isolation_element=matrix_el,
            )
        cnt = {e: sum(1 for o in obs if o.element == e) for e in elements}
        samples.append((label, obs, stark, dict(_TI6AL4V_NOMINAL_WT)))
        print(
            f"  [{gi+1}/{n_groups}] {label}: {len(obs)} obs {cnt}, "
            f"{time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    al_rate = (
        float(np.mean([any(o.element == "Al" for o in s[1]) for s in samples])) if samples else 0.0
    )
    print(f"  Al-measured rate = {al_rate:.2f} ({int(round(al_rate*len(samples)))}/{len(samples)})")

    base_pairs, opc_pairs, prov = opc_heldout(db, samples, closure_mode="standard")
    bsc = score(base_pairs)
    osc = score(opc_pairs)
    print("-" * 74)
    print(f"  baseline (calibration-free) : {_fmt_score(bsc)}")
    print(
        f"  OPC same-comp LOO           : {_fmt_score(osc)}  (selected={prov['n_selected']}, "
        f"robust_T={prov['robust_T_K']:.0f}K)"
    )
    print("  NOTE: single Ti-6Al-4V composition -> OPC LOO holds out the noise/sol")
    print("        realization, the DED known-feedstock scenario (not cross-composition).")
    return {
        "n_observations": len(samples),
        "mode": mode,
        "avg_group": avg_group,
        "al_measured_rate": al_rate,
        "nominal_truth_wt": _TI6AL4V_NOMINAL_WT,
        "baseline": bsc,
        "opc_loo": osc,
        "provenance": prov,
        "per_observation": [
            {"name": n, "baseline_wt": renorm100(_strip(bp)), "opc_wt": renorm100(_strip(op))}
            for (n, _, _, _), (_, bp), (_, op) in zip(samples, base_pairs, opc_pairs)
        ],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bhvo2", action="store_true", help="Part 1: in-repo BHVO-2 (12 spectra).")
    ap.add_argument(
        "--chemcam", action="store_true", help="Part 2: 60-standard ChemCam held-out OPC."
    )
    ap.add_argument(
        "--titanium",
        action="store_true",
        help="Part 3: real Mars SuperCam Ti-6Al-4V (DED matrix) constrained OPC.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap samples (debug).")
    ap.add_argument(
        "--matrix-isolation",
        action="store_true",
        help="Part 3: drop trace lines blended with the dominant matrix (Ti) element.",
    )
    ap.add_argument(
        "--forced",
        action="store_true",
        help="Part 3: constrained force-extraction at KNOWN {Ti,Al,V} line positions "
        "(guarantees Al is measured every spectrum) instead of generic detection.",
    )
    ap.add_argument(
        "--avg-group",
        type=int,
        default=1,
        help="Part 3: average this many consecutive-sol spectra into higher-SNR composites.",
    )
    ap.add_argument("--out", default=None, help="Write the result JSON here.")
    args = ap.parse_args(argv)
    if not (args.bhvo2 or args.chemcam or args.titanium):
        ap.error("pass --bhvo2 and/or --chemcam and/or --titanium")

    import cflibs

    db_path = _db_path()
    print(f"cflibs={Path(cflibs.__file__).resolve().parent}")
    print(f"db={db_path}")
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)

    result: dict = {"db": db_path}
    if args.bhvo2:
        result["bhvo2"] = run_bhvo2(db)
    if args.chemcam:
        result["chemcam"] = run_chemcam(db, args.limit)
    if args.titanium:
        result["titanium"] = run_titanium(
            db,
            args.limit,
            matrix_isolation=args.matrix_isolation,
            forced=args.forced,
            avg_group=args.avg_group,
        )

    out_path = (
        Path(args.out) if args.out else (_REPO_ROOT / "output" / "pds_opc" / "benchmark.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
