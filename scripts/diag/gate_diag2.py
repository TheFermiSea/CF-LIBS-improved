#!/usr/bin/env python3
"""Diagnose (a) why confounders pass comb, (b) the _transition_strength bug's
effect on comb composition, (c) Al recovery under a count-invariant strength."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.io.spectrum import load_spectrum
from cflibs.inversion.identify import line_detection as LD

DB = "/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"
SPEC = "/home/brian/code/CF-LIBS-improved/data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv"
MAJORS = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K", "P"]
CONF = ["Ag", "Sn", "W", "Bi"]
ELEMS = MAJORS + CONF
PRESHIFT = -0.10

wl, inten = load_spectrum(SPEC)
wl = wl + PRESHIFT
db = AtomicDatabase(Path(DB))
peaks = LD._find_peaks(wl, inten, 0.01, 0.2, use_jax_fallback=False)
wl_step = float(np.median(np.diff(wl)))

# --- The unit-mixing bug: show the top-30 comb composition for Al vs Bi ---
trans = LD._load_transitions(db, ELEMS, wl.min(), wl.max(), None)
tbe = {}
for t in trans:
    tbe.setdefault(t.element, []).append(t)

print("=== _transition_strength composition of comb (top-30, floor=None) ===")
for el in ("Al", "Mg", "Bi", "W", "Ag", "Sn"):
    v = tbe.get(el, [])
    comb = LD._select_comb_transitions(v, 30)
    n_zero_relint = sum(1 for t in comb if not (t.relative_intensity and t.relative_intensity > 0))
    n_real_relint = len(comb) - n_zero_relint
    strengths = [LD._transition_strength(t) for t in comb]
    print(f" {el:3}: comb={len(comb):2}  zero-relint(A_ki-fallback)={n_zero_relint:2} "
          f" real-relint={n_real_relint:2}  strength range=[{min(strengths):.2e},{max(strengths):.2e}]")
    # is the bright resonance line (high rel_int) in the comb?
    if el == "Al":
        wls = sorted(round(t.wavelength_nm, 2) for t in comb)
        has396 = any(abs(t.wavelength_nm - 396.152) < 0.02 for t in comb)
        has394 = any(abs(t.wavelength_nm - 394.401) < 0.02 for t in comb)
        print(f"      Al 396.15 in comb? {has396}   Al 394.40 in comb? {has394}")

# --- proposed fix: rel_int only (drop A_ki fallback), how does comb change? ---
def strength_relint_only(t):
    if t.relative_intensity is not None and t.relative_intensity > 0:
        return float(t.relative_intensity)
    return 0.0  # no A_ki fallback

def select_comb_fixed(transitions, max_lines):
    s = sorted(transitions, key=lambda t: (strength_relint_only(t), -t.wavelength_nm), reverse=True)
    return s[:max_lines] if max_lines > 0 and len(s) > max_lines else s

print("\n=== Al comb under FIXED strength (rel_int only) ===")
for el in ("Al", "Mg"):
    comb = select_comb_fixed(tbe.get(el, []), 30)
    has396 = any(abs(t.wavelength_nm - 396.152) < 0.02 for t in comb)
    has394 = any(abs(t.wavelength_nm - 394.401) < 0.02 for t in comb)
    top5 = [(round(t.wavelength_nm, 2), t.relative_intensity) for t in comb[:5]]
    print(f" {el}: 396.15in={has396} 394.40in={has394}  top5(wl,relint)={top5}")

# --- WHY do confounders pass comb? show their comb match stats at best shift ---
print("\n=== confounder comb scores (floor=None, no kdet) ===")
comb_tbe = {e: LD._select_comb_transitions(v, 30) for e, v in tbe.items()}
grid = LD._build_shift_grid(0.5, None, wl_step, 0.1)
best, fb = LD._scan_comb_shifts(
    peaks=peaks, transitions_by_element=comb_tbe, shift_grid=grid,
    total_peaks=len(peaks), wavelength_tolerance_nm=0.1,
    comb_min_matches=3, comb_min_precision=0.02, comb_min_recall=0.1,
    comb_max_missing_fraction=0.85,
)
scores = best["scores"] if best else fb["scores"]
print(f" best shift={best['shift_nm'] if best else 'NA':}")
for el in CONF + ["Al", "Mg", "Na", "Ca", "Fe"]:
    s = scores.get(el)
    if s:
        print(f"  {el:3}: matched={s.matched_lines:3} expected={s.expected_lines:3} "
              f"prec={s.precision:.3f} recall={s.recall:.3f} miss={s.missing_fraction:.3f} pass={s.passes}")
