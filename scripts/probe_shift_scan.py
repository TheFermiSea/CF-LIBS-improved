#!/usr/bin/env python3
"""Trace the global shift-scan objective: WHY does it pick the shift it picks?

Reproduces detect_line_observations' internal comb shift-scan for the
production config and prints, per shift: total_f1, total_matches_pass,
passed_elements, and per-element matched counts for the real majors. Also
reports what applied_shift_nm the pipeline actually selects, and how many of
Al's 4 resonance lines align at that shift vs. at the true-offset shift.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.io.spectrum import load_spectrum
from cflibs.benchmark.reference_compositions import BHVO2_BASALT_USGS
import cflibs.inversion.identify.line_detection as LD

ROOT = Path("/home/brian/code/CF-LIBS-improved")
spec = sys.argv[1] if len(sys.argv) > 1 else "data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv"
sp = Path(spec)
if not sp.exists():
    sp = ROOT / spec

wl, inten = load_spectrum(str(sp))
db = AtomicDatabase(ROOT / "ASD_da/libs_production.db")
print(f"spectrum: {sp.name}  n={len(wl)}  range {wl.min():.1f}-{wl.max():.1f} nm  "
      f"dl={np.median(np.diff(wl)):.4f}")

elements = list(BHVO2_BASALT_USGS.keys()) + ["Ag", "Sn", "W", "Bi"]
print(f"requested elements: {elements}")
print(f"HAS_RUST_CORE={LD.HAS_RUST_CORE}")

# Production config defaults (CLI _detect_and_select_lines / detect_line_observations)
WL_TOL = 0.1
MIN_PEAK_H = 0.01
PEAK_W = 0.2
MIN_REL_INT = 100.0
SHIFT_SCAN = 0.5
COMB_MAX = 30
COMB_MIN_MATCHES = 3
COMB_MIN_PREC = 0.02
COMB_MIN_RECALL = 0.1
COMB_MAX_MISS = 0.85

wl_step = float(np.median(np.diff(wl)))
peaks = LD._find_peaks(wl, inten, MIN_PEAK_H, PEAK_W, use_jax_fallback=False)
total_peaks = len(peaks)
print(f"total peaks detected: {total_peaks}")

transitions = LD._load_transitions(
    db, elements, wavelength_min=float(wl.min()), wavelength_max=float(wl.max()),
    min_relative_intensity=MIN_REL_INT,
)
tbe = {}
for t in transitions:
    tbe.setdefault(t.element, []).append(t)
print(f"transitions loaded (rel_int>={MIN_REL_INT}): "
      f"{ {e: len(v) for e, v in tbe.items()} }")

# kdet filter (as in production)
filt, kw = LD._kdet_filter_elements(
    peaks=peaks, transitions_by_element=tbe, shift_scan_nm=SHIFT_SCAN,
    shift_step_nm=None, wavelength_tolerance_nm=WL_TOL, wl_step=wl_step,
    kdet_min_score=0.05, kdet_min_candidates=2, kdet_rarity_power=0.5,
    kdet_weight_clip=(0.25, 4.0), use_jax=False,
)
print(f"kdet survivors: {sorted(filt.keys())}  warnings={kw}")
tbe = filt if filt else tbe

comb_tbe = {e: LD._select_comb_transitions(v, COMB_MAX) for e, v in tbe.items()}
shift_grid = LD._build_shift_grid(SHIFT_SCAN, None, wl_step, WL_TOL)
print(f"shift grid: {len(shift_grid)} pts, "
      f"[{shift_grid.min():+.3f}, {shift_grid.max():+.3f}] step~{np.median(np.diff(shift_grid)):.4f}")

# Force the PYTHON scan path so we can introspect per-shift (Rust returns only best)
def py_scan(shift_grid):
    rows = []
    for shift_nm in shift_grid:
        scores = {}
        total_f1 = 0.0
        total_matches_pass = 0
        total_matches_all = 0
        passed = []
        for el, trans in comb_tbe.items():
            if not trans:
                continue
            s = LD._score_comb_for_element(
                peaks=peaks, transitions=trans, shift_nm=float(shift_nm),
                total_peaks=total_peaks, wavelength_tolerance_nm=WL_TOL,
                comb_min_matches=COMB_MIN_MATCHES, comb_min_precision=COMB_MIN_PREC,
                comb_min_recall=COMB_MIN_RECALL, comb_max_missing_fraction=COMB_MAX_MISS,
            )
            s.element = el
            scores[el] = s
            total_matches_all += s.matched_lines
            if s.passes:
                passed.append(el)
                total_f1 += s.f1_score
                total_matches_pass += s.matched_lines
        rows.append({
            "shift": float(shift_nm), "f1": total_f1,
            "mpass": total_matches_pass, "mall": total_matches_all,
            "passed": passed, "scores": scores,
        })
    return rows

rows = py_scan(shift_grid)

# Replicate the actual best-selection logic from _scan_comb_shifts
best = None
for r in rows:
    if best is None:
        best = r
        continue
    better = False
    if r["f1"] > best["f1"]:
        better = True
    elif np.isclose(r["f1"], best["f1"]):
        if r["mpass"] > best["mpass"]:
            better = True
        elif r["mpass"] == best["mpass"] and abs(r["shift"]) < abs(best["shift"]):
            better = True
    if better:
        best = r
print(f"\n*** SELECTED applied_shift_nm = {best['shift']:+.4f}  "
      f"(total_f1={best['f1']:.4f}, matches_pass={best['mpass']}, "
      f"passed={best['passed']}) ***")

# Top shifts by F1
print("\nTop 12 shifts by total_f1:")
print(f"  {'shift':>8}{'f1':>9}{'m_pass':>8}{'m_all':>7}  passed_elements")
for r in sorted(rows, key=lambda x: (-x["f1"], -x["mpass"], abs(x["shift"])))[:12]:
    print(f"  {r['shift']:>+8.3f}{r['f1']:>9.4f}{r['mpass']:>8}{r['mall']:>7}  {r['passed']}")

# Focused look near the true offset (peaks at true+0.10 => align shift ~ -0.10)
print("\nObjective curve near interesting shifts (per-element matched counts):")
focus_targets = [-0.20, -0.15, -0.10, -0.05, 0.0, +0.05, +0.10]
maj = ["Mg", "Ca", "Si", "Fe", "Al", "Na", "Ti", "K", "Mn", "P"]
hdr = "  " + f"{'shift':>7}{'f1':>8}{'m_all':>7}  " + "".join(f"{m:>4}" for m in maj)
print(hdr)
for tgt in focus_targets:
    r = min(rows, key=lambda x: abs(x["shift"] - tgt))
    cells = "".join(f"{r['scores'].get(m).matched_lines if m in r['scores'] else 0:>4}" for m in maj)
    print(f"  {r['shift']:>+7.3f}{r['f1']:>8.4f}{r['mall']:>7}  {cells}")

# How many of Al's 4 resonance lines align at selected vs true-offset shift?
al_reson = [308.215, 309.271, 394.401, 396.152]
peak_wl = np.array([p[1] for p in peaks])
def al_aligned(shift):
    n = 0
    details = []
    for line in al_reson:
        if line < wl.min() or line > wl.max():
            continue
        d = np.abs((peak_wl + shift) - line)
        j = int(np.argmin(d))
        ok = d[j] <= WL_TOL
        n += int(ok)
        details.append((line, float(d[j]), ok))
    return n, details

for label, shift in [("SELECTED", best["shift"]), ("true-align -0.10", -0.10),
                     ("zero", 0.0)]:
    n, det = al_aligned(shift)
    print(f"\nAl lines within tol={WL_TOL} at shift={shift:+.3f} [{label}]: {n}/4")
    for line, d, ok in det:
        print(f"    {line:.3f}  delta_after_shift={d:.4f}  {'OK' if ok else 'miss'}")
