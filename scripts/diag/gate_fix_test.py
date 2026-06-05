#!/usr/bin/env python3
"""Test proposed robust gate settings on aligned ChemCam: recover majors,
keep confounders out, without re-detonating Na Rydberg."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.io.spectrum import load_spectrum
from cflibs.inversion.identify.line_detection import detect_line_observations

DB = "/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"
SPEC = "/home/brian/code/CF-LIBS-improved/data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv"
MAJORS = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K", "P"]
CONF = ["Ag", "Sn", "W", "Bi"]
ELEMS = MAJORS + CONF
wl, inten = load_spectrum(SPEC)
wl = wl - 0.10
db = AtomicDatabase(Path(DB))


def run(label, **kw):
    res = detect_line_observations(wl, inten, db, ELEMS, **kw)
    by = {}
    for o in res.observations:
        by[o.element] = by.get(o.element, 0) + 1
    msurv = [e for e in MAJORS if by.get(e)]
    mdrop = [e for e in MAJORS if not by.get(e)]
    csurv = {e: by[e] for e in CONF if by.get(e)}
    print(f"\n[{label}]")
    print(f"  majors w/ obs: { {e: by.get(e, 0) for e in MAJORS} }")
    print(f"  majors DROPPED: {mdrop}")
    print(f"  confounders present: {csurv or 'NONE'}  n_obs={len(res.observations)} warns={res.warnings}")


# baseline (production defaults)
run("PROD floor=100 kdet=on comb default", min_relative_intensity=100.0)

# A: lower floor to 1 (keep zero/near-zero out, admit Mg/K/Na real lines)
run("A floor=1", min_relative_intensity=1.0)

# B: floor=1 + tighten comb precision + lower missing fraction (reject confounders)
run("B floor=1 comb prec=0.06 miss=0.6 recall=0.2",
    min_relative_intensity=1.0, comb_min_precision=0.06,
    comb_max_missing_fraction=0.6, comb_min_recall=0.2)

# C: floor=1 + kdet off + tightened comb
run("C floor=1 kdet=off comb prec=0.06 miss=0.6 recall=0.2",
    min_relative_intensity=1.0, kdet_enabled=False, comb_min_precision=0.06,
    comb_max_missing_fraction=0.6, comb_min_recall=0.2)

# D: floor=1 + comb_min_matches=5 (count-variant, for contrast)
run("D floor=1 kdet=off comb_min_matches=5 prec=0.08",
    min_relative_intensity=1.0, kdet_enabled=False, comb_min_matches=5,
    comb_min_precision=0.08, comb_max_missing_fraction=0.6, comb_min_recall=0.2)
