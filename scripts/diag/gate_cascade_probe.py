#!/usr/bin/env python3
"""Detection-cascade gate audit on a CORRECTLY-ALIGNED ChemCam BHVO-2 spectrum.

For each gate from 'transitions loaded' -> 'observation reaches solver', report
which of the 10 cert majors survive and why dropped. Alignment: ChemCam has a
constant +0.10 nm offset (DB lines sit 0.10 nm below observed peaks); we apply
preshift -0.10 to the OBSERVED wavelength axis so DB lines align.
"""
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
PRESHIFT = -0.10  # align observed axis to DB

wl, inten = load_spectrum(SPEC)
wl = wl + PRESHIFT
db = AtomicDatabase(Path(DB))


def counts_by_el(trans_by_el):
    return {e: len(v) for e, v in trans_by_el.items() if v}


print("=" * 78)
print("GATE CASCADE on ALIGNED ChemCam (preshift %.2f nm)" % PRESHIFT)
print("=" * 78)

# ---- GATE 0: SQL rel_int floor (get_transitions) ----
for floor in (None, 100.0):
    trans = LD._load_transitions(db, ELEMS, wl.min(), wl.max(), floor)
    by_el = {}
    for t in trans:
        by_el.setdefault(t.element, []).append(t)
    surv = [e for e in MAJORS if by_el.get(e)]
    drop = [e for e in MAJORS if not by_el.get(e)]
    print(f"\n[GATE 0 rel_int floor={floor}] majors with >=1 candidate transition:")
    print(f"   counts: { {e: len(by_el.get(e, [])) for e in MAJORS} }")
    print(f"   DROPPED majors (0 transitions): {drop}")

# Use floor=None from here so we can isolate downstream gates
floor = 100.0
trans = LD._load_transitions(db, ELEMS, wl.min(), wl.max(), floor)
by_el = {}
for t in trans:
    by_el.setdefault(t.element, []).append(t)

# ---- peaks ----
peaks = LD._find_peaks(wl, inten, 0.01, 0.2, use_jax_fallback=False)
print(f"\nPeaks detected: {len(peaks)}")
wl_step = float(np.median(np.diff(wl)))

# ---- GATE 1: kdet ----
for floor in (None, 100.0):
    trans = LD._load_transitions(db, ELEMS, wl.min(), wl.max(), floor)
    tbe = {}
    for t in trans:
        tbe.setdefault(t.element, []).append(t)
    filt, warns = LD._kdet_filter_elements(
        peaks=peaks, transitions_by_element=tbe,
        shift_scan_nm=0.5, shift_step_nm=None,
        wavelength_tolerance_nm=0.1, wl_step=wl_step,
        kdet_min_score=0.05, kdet_min_candidates=2,
        kdet_rarity_power=0.5, kdet_weight_clip=(0.25, 4.0), use_jax=False,
    )
    surv = [e for e in MAJORS if filt.get(e)]
    drop = [e for e in MAJORS if e in tbe and not filt.get(e)]
    confsurv = [e for e in CONF if filt.get(e)]
    print(f"\n[GATE 1 kdet, floor={floor}] majors surviving kdet: {surv}")
    print(f"   majors KILLED by kdet (had transitions, dropped): {drop}")
    print(f"   confounders surviving kdet: {confsurv}  warns={warns}")

# ---- GATE 2: comb (use floor=None, kdet output) ----
for floor in (None, 100.0):
    trans = LD._load_transitions(db, ELEMS, wl.min(), wl.max(), floor)
    tbe = {}
    for t in trans:
        tbe.setdefault(t.element, []).append(t)
    filt, _ = LD._kdet_filter_elements(
        peaks=peaks, transitions_by_element=tbe, shift_scan_nm=0.5,
        shift_step_nm=None, wavelength_tolerance_nm=0.1, wl_step=wl_step,
        kdet_min_score=0.05, kdet_min_candidates=2, kdet_rarity_power=0.5,
        kdet_weight_clip=(0.25, 4.0), use_jax=False,
    )
    comb_tbe = {e: LD._select_comb_transitions(v, 30) for e, v in filt.items()}
    grid = LD._build_shift_grid(0.5, None, wl_step, 0.1)
    best, fb = LD._scan_comb_shifts(
        peaks=peaks, transitions_by_element=comb_tbe, shift_grid=grid,
        total_peaks=len(peaks), wavelength_tolerance_nm=0.1,
        comb_min_matches=3, comb_min_precision=0.02, comb_min_recall=0.1,
        comb_max_missing_fraction=0.85,
    )
    if best and best["passed_elements"]:
        passed = list(best["passed_elements"])
        shift = best["shift_nm"]
    else:
        passed = []
        shift = fb["shift_nm"] if fb else 0.0
    msurv = [e for e in MAJORS if e in passed]
    mdrop = [e for e in MAJORS if e in filt and e not in passed]
    csurv = [e for e in CONF if e in passed]
    print(f"\n[GATE 2 comb, floor={floor}] best shift={shift:+.3f} majors passing comb: {msurv}")
    print(f"   majors KILLED by comb: {mdrop}")
    print(f"   confounders passing comb: {csurv}")
