#!/usr/bin/env python3
"""Localize WHERE element over-attribution enters the CF-LIBS closure.

Captures the closure inputs (per-element Boltzmann intercept q_s, partition
function U_s(T), abundance multiplier) and shows the closure term
C_s ∝ U_s·exp(q_s)·mult vs cert. Measured finding (ChemCam BHVO-2):
the intercepts q are nearly FLAT (~20) across elements, so the dominance is
driven by U_Fe=50 / U_Ti=64; physically q_Si should sit ~3 above q_Fe.
Self-absorption correction makes it WORSE (Fe 72->89%). Run to reproduce.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.reference_compositions import BHVO2_BASALT_USGS as CERT
from cflibs.cli.main import _detect_and_select_lines
from cflibs.io.spectrum import load_spectrum
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
from cflibs.inversion.physics import closure as cm

spec = sys.argv[1] if len(sys.argv) > 1 else "chemcam_bhvo2_loc1"
sa = "--sa" in sys.argv
cert = dict(CERT); cert_el = list(cert)
db = AtomicDatabase(Path("/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"))
wl, it = load_spectrum(f"/home/brian/code/CF-LIBS-improved/data/bhvo2_usgs/{spec}_spectrum.csv")
obs = _detect_and_select_lines(wl, it, db, cert_el, apply_self_absorption=sa)

orig = cm.ClosureEquation.apply_standard
cap: dict = {}
def spy(intercepts, partition_funcs, abundance_multipliers=None):
    cap["q"] = dict(intercepts); cap["U"] = dict(partition_funcs)
    cap["m"] = dict(abundance_multipliers) if abundance_multipliers else {}
    return orig(intercepts, partition_funcs, abundance_multipliers)
cm.ClosureEquation.apply_standard = staticmethod(spy)

res = IterativeCFLIBSSolver(atomic_db=db, apply_self_absorption=sa).solve(obs)
pred = dict(res.concentrations)
q, U, m = cap.get("q", {}), cap.get("U", {}), cap.get("m", {})
nobs: dict = {}
for o in obs:
    nobs[getattr(o, "element", None)] = nobs.get(getattr(o, "element", None), 0) + 1
print(f"{spec}  SA={sa}  T={res.temperature_K:.0f}K  ne={res.electron_density_cm3:.2e}")
print(f"{'el':<4}{'q':>12}{'U':>9}{'mult':>9}{'U*exp(q)*m':>13}{'C%':>8}{'cert%':>8}{'nobs':>5}")
for e in sorted(set(list(q) + cert_el)):
    qi = q.get(e, float("nan")); Ui = U.get(e, float("nan")); mi = m.get(e, 1.0)
    term = (Ui * math.exp(qi) * mi) if (qi == qi and Ui == Ui) else float("nan")
    print(f"{e:<4}{qi:>12.3f}{Ui:>9.2f}{mi:>9.3f}{term:>13.3e}"
          f"{pred.get(e,0)*100:>8.2f}{cert.get(e,0)*100:>8.2f}{nobs.get(e,0):>5}")
