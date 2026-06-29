#!/usr/bin/env python3
"""Validated Saha-Boltzmann GRAPH method probe (CD-SB graph, Aguilera & Aragon).

Pools ALL detected lines of ALL species/stages onto ONE graph with a single
shared slope (-1/kT). Ion lines are shifted onto the neutral plane:
  neutral (z=1): x = E_k,           y = ln(I*wl/(g_k*A_ki))
  ion     (z>1): x = E_k + IP1*(z-1), y = ln(I*wl/(g_k*A_ki)) - ln(S)*(z-1)
  with ln(S) = ln(SAHA_CONST_CM3/n_e * T_eV**1.5)  [ion->neutral Saha shift]
A global least-squares fit (shared slope + per-element intercept dummies)
yields each element's neutral intercept = ln(n_I/U_I); total element density
n_tot = n_I*(1 + n_II/n_I) via Saha. Composition = mass-fraction(n_tot).

Why this beats per-element neutral-only Boltzmann: Fe/Ti ION lines sit at
x = E_k+IP ~ 15-23 eV, giving a long lever arm that well-conditions the
intercept (neutral-only Fe spans only ~1 eV -> unstable, R^2~0).

MEASURED (2026-06-05) on real BHVO-2 ChemCam loc1:
  SB-graph standard:  RMSE 8.11 (Fe 39->18), global R^2 0.945  [vs current 10.33]
  (still beaten by oxide closure 5.60; target = SB-graph + oxide combined)
Per-species CDSBPlotter (WRONG tool) gave RMSE 30 (per-species fits fail).
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.reference_compositions import BHVO2_BASALT_USGS as CERT
from cflibs.cli.main import _detect_and_select_lines
from cflibs.io.spectrum import load_spectrum
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
from cflibs.core.constants import SAHA_CONST_CM3

ATOM = {"Si":28.09,"Ti":47.87,"Al":26.98,"Fe":55.85,"Mn":54.94,"Mg":24.31,
        "Ca":40.08,"Na":22.99,"K":39.10,"P":30.97}
IP1 = {"Si":8.15,"Ti":6.83,"Al":5.99,"Fe":7.90,"Mn":7.43,"Mg":7.65,
       "Ca":6.11,"Na":5.14,"K":4.34,"P":10.49}


def sb_graph_composition(obs, db, T, ne):
    T_eV = T / 11604.5
    lnS = math.log(SAHA_CONST_CM3 / ne * T_eV ** 1.5)

    def U(el, z):
        p = db.partition_function_for(el, z)
        return float(p.at(T)) if p else (25.0 if z == 1 else 15.0)

    els = sorted({o.element for o in obs})
    elidx = {e: i for i, e in enumerate(els)}
    rows = []
    for o in obs:
        if o.A_ki <= 0 or o.intensity <= 0:
            continue
        y = math.log(o.intensity * o.wavelength_nm / (o.g_k * o.A_ki))
        z = o.ionization_stage
        x = o.E_k_ev if z == 1 else o.E_k_ev + IP1.get(o.element, 8.0) * (z - 1)
        if z > 1:
            y -= lnS * (z - 1)
        rows.append((o.element, x, y))
    A = np.zeros((len(rows), 1 + len(els)))
    yv = np.zeros(len(rows))
    for k, (e, x, y) in enumerate(rows):
        A[k, 0] = x
        A[k, 1 + elidx[e]] = 1.0
        yv[k] = y
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    slope = coef[0]
    icpt = {e: coef[1 + elidx[e]] for e in els}
    pr = A @ coef
    r2 = 1 - np.sum((yv - pr) ** 2) / np.sum((yv - yv.mean()) ** 2)
    ntot = {}
    for e in els:
        if e not in IP1:
            continue
        s = (SAHA_CONST_CM3 / ne) * (T_eV ** 1.5) * (U(e, 2) / U(e, 1)) * math.exp(-IP1[e] / T_eV)
        ntot[e] = U(e, 1) * math.exp(icpt[e]) * (1 + s)
    return ntot, slope, r2


def main():
    cert = dict(CERT)
    cert_el = list(cert)
    db = AtomicDatabase(Path("/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"))
    for spec in ["chemcam_bhvo2_loc1", "csa_bhvo2_1000pulse"]:
        wl, it = load_spectrum(f"/home/brian/code/CF-LIBS-improved/data/bhvo2_usgs/{spec}_spectrum.csv")
        obs = _detect_and_select_lines(wl, it, db, cert_el)
        res0 = IterativeCFLIBSSolver(atomic_db=db).solve(obs)
        ntot, slope, r2 = sb_graph_composition(obs, db, res0.temperature_K, res0.electron_density_cm3)
        ms = sum(ntot[e] * ATOM[e] for e in ntot)
        pred = {e: ntot[e] * ATOM[e] / ms for e in ntot}
        rmse = math.sqrt(sum((pred.get(e, 0) - cert[e]) ** 2 for e in cert_el) / len(cert_el)) * 100
        Tfit = -1 / (slope / 11604.5) if slope < 0 else float("inf")
        top = sorted(((pred.get(e, 0) * 100, e) for e in cert_el), reverse=True)[:5]
        print(f"{spec}: SB-graph RMSE={rmse:.2f} Tfit={Tfit:.0f}K R2={r2:.3f}  " +
              " ".join(f"{e}={v:.0f}" for v, e in top))


if __name__ == "__main__":
    main()
